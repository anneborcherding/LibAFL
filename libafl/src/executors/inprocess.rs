//! The [`InProcessExecutor`] is a libfuzzer-like executor, that will simply call a function.
//! It should usually be paired with extra error-handling, such as a restarting event manager, to be effective.
//!
//! Needs the `fork` feature flag.
#![allow(clippy::needless_pass_by_value)]

use alloc::boxed::Box;
#[cfg(all(unix, feature = "std"))]
use alloc::vec::Vec;
#[cfg(all(feature = "std", unix, target_os = "linux"))]
use core::ptr::addr_of_mut;
#[cfg(all(unix, feature = "std"))]
use core::time::Duration;
use core::{
    borrow::BorrowMut,
    ffi::c_void,
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    ptr::{self, null_mut},
};
#[cfg(any(unix, all(windows, feature = "std")))]
use core::{
    ptr::write_volatile,
    sync::atomic::{compiler_fence, Ordering},
};
#[cfg(all(feature = "std", unix))]
use std::intrinsics::transmute;

#[cfg(all(unix, not(miri)))]
use libafl_bolts::os::unix_signals::setup_signal_handler;
#[cfg(all(feature = "std", unix))]
use libafl_bolts::os::unix_signals::{ucontext_t, Handler, Signal};
#[cfg(all(windows, feature = "std"))]
use libafl_bolts::os::windows_exceptions::setup_exception_handler;
#[cfg(all(feature = "std", unix))]
use libafl_bolts::shmem::ShMemProvider;
#[cfg(all(feature = "std", unix))]
use libc::siginfo_t;
#[cfg(all(feature = "std", unix))]
use nix::{
    sys::wait::{waitpid, WaitStatus},
    unistd::{fork, ForkResult},
};
#[cfg(windows)]
use windows::Win32::System::Threading::SetThreadStackGuarantee;
#[cfg(all(windows, feature = "std"))]
use windows::Win32::System::Threading::PTP_TIMER;

use crate::{
    events::{EventFirer, EventRestarter},
    executors::{Executor, ExitKind, HasObservers},
    feedbacks::Feedback,
    fuzzer::HasObjective,
    inputs::UsesInput,
    observers::{ObserversTuple, UsesObservers},
    state::{HasClientPerfMonitor, HasCorpus, HasExecutions, HasSolutions, UsesState},
    Error,
};

/// The process executor simply calls a target function, as mutable reference to a closure
pub type InProcessExecutor<'a, H, OT, S> = GenericInProcessExecutor<H, &'a mut H, OT, S>;

/// The process executor simply calls a target function, as boxed `FnMut` trait object
pub type OwnedInProcessExecutor<OT, S> = GenericInProcessExecutor<
    dyn FnMut(&<S as UsesInput>::Input) -> ExitKind,
    Box<dyn FnMut(&<S as UsesInput>::Input) -> ExitKind>,
    OT,
    S,
>;

/// The inmem executor simply calls a target function, then returns afterwards.
#[allow(dead_code)]
pub struct GenericInProcessExecutor<H, HB, OT, S>
where
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    HB: BorrowMut<H>,
    OT: ObserversTuple<S>,
    S: UsesInput,
{
    /// The harness function, being executed for each fuzzing loop execution
    harness_fn: HB,
    /// The observers, observing each run
    observers: OT,
    // Crash and timeout hah
    handlers: InProcessHandlers,
    phantom: PhantomData<(S, *const H)>,
}

impl<H, HB, OT, S> Debug for GenericInProcessExecutor<H, HB, OT, S>
where
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    HB: BorrowMut<H>,
    OT: ObserversTuple<S>,
    S: UsesInput,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("GenericInProcessExecutor")
            .field("harness_fn", &"<fn>")
            .field("observers", &self.observers)
            .finish_non_exhaustive()
    }
}

impl<H, HB, OT, S> UsesState for GenericInProcessExecutor<H, HB, OT, S>
where
    H: ?Sized + FnMut(&S::Input) -> ExitKind,
    HB: BorrowMut<H>,
    OT: ObserversTuple<S>,
    S: UsesInput,
{
    type State = S;
}

impl<H, HB, OT, S> UsesObservers for GenericInProcessExecutor<H, HB, OT, S>
where
    H: ?Sized + FnMut(&S::Input) -> ExitKind,
    HB: BorrowMut<H>,
    OT: ObserversTuple<S>,
    S: UsesInput,
{
    type Observers = OT;
}

impl<EM, H, HB, OT, S, Z> Executor<EM, Z> for GenericInProcessExecutor<H, HB, OT, S>
where
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    HB: BorrowMut<H>,
    EM: UsesState<State = S>,
    OT: ObserversTuple<S>,
    S: UsesInput + HasExecutions,
    Z: UsesState<State = S>,
{
    fn run_target(
        &mut self,
        fuzzer: &mut Z,
        state: &mut Self::State,
        mgr: &mut EM,
        input: &Self::Input,
    ) -> Result<ExitKind, Error> {
        *state.executions_mut() += 1;
        self.handlers
            .pre_run_target(self, fuzzer, state, mgr, input);

        let ret = (self.harness_fn.borrow_mut())(input);

        self.handlers.post_run_target();
        Ok(ret)
    }
}

impl<H, HB, OT, S> HasObservers for GenericInProcessExecutor<H, HB, OT, S>
where
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    HB: BorrowMut<H>,
    OT: ObserversTuple<S>,
    S: UsesInput,
{
    #[inline]
    fn observers(&self) -> &OT {
        &self.observers
    }

    #[inline]
    fn observers_mut(&mut self) -> &mut OT {
        &mut self.observers
    }
}

impl<H, HB, OT, S> GenericInProcessExecutor<H, HB, OT, S>
where
    H: FnMut(&<S as UsesInput>::Input) -> ExitKind + ?Sized,
    HB: BorrowMut<H>,
    OT: ObserversTuple<S>,
    S: HasSolutions + HasClientPerfMonitor + HasCorpus,
{
    /// Create a new in mem executor.
    /// Caution: crash and restart in one of them will lead to odd behavior if multiple are used,
    /// depending on different corpus or state.
    /// * `harness_fn` - the harness, executing the function
    /// * `observers` - the observers observing the target during execution
    /// This may return an error on unix, if signal handler setup fails
    pub fn new<EM, OF, Z>(
        harness_fn: HB,
        observers: OT,
        _fuzzer: &mut Z,
        _state: &mut S,
        _event_mgr: &mut EM,
    ) -> Result<Self, Error>
    where
        Self: Executor<EM, Z, State = S>,
        EM: EventFirer<State = S> + EventRestarter,
        OF: Feedback<S>,
        Z: HasObjective<Objective = OF, State = S>,
    {
        let handlers = InProcessHandlers::new::<Self, EM, OF, Z>()?;
        #[cfg(windows)]
        // Some initialization necessary for windows.
        unsafe {
            /*
                See https://github.com/AFLplusplus/LibAFL/pull/403
                This one reserves certain amount of memory for the stack.
                If stack overflow happens during fuzzing on windows, the program is transferred to our exception handler for windows.
                However, if we run out of the stack memory again in this exception handler, we'll crash with STATUS_ACCESS_VIOLATION.
                We need this API call because with the llmp_compression
                feature enabled, the exception handler uses a lot of stack memory (in the compression lib code) on release build.
                As far as I have observed, the compression uses around 0x10000 bytes, but for safety let's just reserve 0x20000 bytes for our exception handlers.
                This number 0x20000 could vary depending on the compilers optimization for future compression library changes.
            */
            let mut stack_reserved = 0x20000;
            SetThreadStackGuarantee(&mut stack_reserved)?;
        }
        Ok(Self {
            harness_fn,
            observers,
            handlers,
            phantom: PhantomData,
        })
    }

    /// Retrieve the harness function.
    #[inline]
    pub fn harness(&self) -> &H {
        self.harness_fn.borrow()
    }

    /// Retrieve the harness function for a mutable reference.
    #[inline]
    pub fn harness_mut(&mut self) -> &mut H {
        self.harness_fn.borrow_mut()
    }

    /// The inprocess handlers
    #[inline]
    pub fn handlers(&self) -> &InProcessHandlers {
        &self.handlers
    }

    /// The inprocess handlers (mutable)
    #[inline]
    pub fn handlers_mut(&mut self) -> &mut InProcessHandlers {
        &mut self.handlers
    }
}

/// The struct has [`InProcessHandlers`].
#[cfg(windows)]
pub trait HasInProcessHandlers {
    /// Get the in-process handlers.
    fn inprocess_handlers(&self) -> &InProcessHandlers;
}

#[cfg(windows)]
impl<H, HB, OT, S> HasInProcessHandlers for GenericInProcessExecutor<H, HB, OT, S>
where
    H: FnMut(&<S as UsesInput>::Input) -> ExitKind + ?Sized,
    HB: BorrowMut<H>,
    OT: ObserversTuple<S>,
    S: HasSolutions + HasClientPerfMonitor + HasCorpus,
{
    /// the timeout handler
    #[inline]
    fn inprocess_handlers(&self) -> &InProcessHandlers {
        &self.handlers
    }
}

/// The inmem executor's handlers.
#[derive(Debug)]
pub struct InProcessHandlers {
    /// On crash C function pointer
    #[cfg(any(unix, feature = "std"))]
    pub crash_handler: *const c_void,
    /// On timeout C function pointer
    #[cfg(any(unix, feature = "std"))]
    pub timeout_handler: *const c_void,
}

impl InProcessHandlers {
    /// Call before running a target.
    #[allow(clippy::unused_self)]
    pub fn pre_run_target<E, EM, I, S, Z>(
        &self,
        _executor: &E,
        _fuzzer: &mut Z,
        _state: &mut S,
        _mgr: &mut EM,
        _input: &I,
    ) {
        #[cfg(unix)]
        unsafe {
            let data = &mut GLOBAL_STATE;
            write_volatile(
                &mut data.current_input_ptr,
                _input as *const _ as *const c_void,
            );
            write_volatile(
                &mut data.executor_ptr,
                _executor as *const _ as *const c_void,
            );
            data.crash_handler = self.crash_handler;
            data.timeout_handler = self.timeout_handler;
            // Direct raw pointers access /aliasing is pretty undefined behavior.
            // Since the state and event may have moved in memory, refresh them right before the signal may happen
            write_volatile(&mut data.state_ptr, _state as *mut _ as *mut c_void);
            write_volatile(&mut data.event_mgr_ptr, _mgr as *mut _ as *mut c_void);
            write_volatile(&mut data.fuzzer_ptr, _fuzzer as *mut _ as *mut c_void);
            compiler_fence(Ordering::SeqCst);
        }
        #[cfg(all(windows, feature = "std"))]
        unsafe {
            let data = &mut GLOBAL_STATE;
            write_volatile(
                &mut data.current_input_ptr,
                _input as *const _ as *const c_void,
            );
            write_volatile(
                &mut data.executor_ptr,
                _executor as *const _ as *const c_void,
            );
            data.crash_handler = self.crash_handler;
            data.timeout_handler = self.timeout_handler;
            // Direct raw pointers access /aliasing is pretty undefined behavior.
            // Since the state and event may have moved in memory, refresh them right before the signal may happen
            write_volatile(&mut data.state_ptr, _state as *mut _ as *mut c_void);
            write_volatile(&mut data.event_mgr_ptr, _mgr as *mut _ as *mut c_void);
            write_volatile(&mut data.fuzzer_ptr, _fuzzer as *mut _ as *mut c_void);
            compiler_fence(Ordering::SeqCst);
        }
    }

    /// Call after running a target.
    #[allow(clippy::unused_self)]
    pub fn post_run_target(&self) {
        #[cfg(unix)]
        unsafe {
            write_volatile(&mut GLOBAL_STATE.current_input_ptr, ptr::null());
            compiler_fence(Ordering::SeqCst);
        }
        #[cfg(all(windows, feature = "std"))]
        unsafe {
            write_volatile(&mut GLOBAL_STATE.current_input_ptr, ptr::null());
            compiler_fence(Ordering::SeqCst);
        }
    }

    /// Create new [`InProcessHandlers`].
    #[cfg(not(all(windows, feature = "std")))]
    pub fn new<E, EM, OF, Z>() -> Result<Self, Error>
    where
        E: Executor<EM, Z> + HasObservers,
        EM: EventFirer<State = E::State> + EventRestarter<State = E::State>,
        OF: Feedback<E::State>,
        E::State: HasSolutions + HasClientPerfMonitor + HasCorpus,
        Z: HasObjective<Objective = OF, State = E::State>,
    {
        #[cfg(unix)]
        #[cfg_attr(miri, allow(unused_variables))]
        unsafe {
            let data = &mut GLOBAL_STATE;
            #[cfg(feature = "std")]
            unix_signal_handler::setup_panic_hook::<E, EM, OF, Z>();
            #[cfg(not(miri))]
            setup_signal_handler(data)?;
            compiler_fence(Ordering::SeqCst);
            Ok(Self {
                crash_handler: unix_signal_handler::inproc_crash_handler::<E, EM, OF, Z>
                    as *const c_void,
                timeout_handler: unix_signal_handler::inproc_timeout_handler::<E, EM, OF, Z>
                    as *const _,
            })
        }
        #[cfg(not(any(unix, feature = "std")))]
        Ok(Self {})
    }

    /// Create new [`InProcessHandlers`].
    #[cfg(all(windows, feature = "std"))]
    pub fn new<E, EM, OF, Z>() -> Result<Self, Error>
    where
        E: Executor<EM, Z> + HasObservers + HasInProcessHandlers,
        EM: EventFirer<State = E::State> + EventRestarter<State = E::State>,
        OF: Feedback<E::State>,
        E::State: HasSolutions + HasClientPerfMonitor + HasCorpus,
        Z: HasObjective<Objective = OF, State = E::State>,
    {
        unsafe {
            let data = &mut GLOBAL_STATE;
            #[cfg(feature = "std")]
            windows_exception_handler::setup_panic_hook::<E, EM, OF, Z>();
            setup_exception_handler(data)?;
            compiler_fence(Ordering::SeqCst);

            Ok(Self {
                crash_handler: windows_exception_handler::inproc_crash_handler::<E, EM, OF, Z>
                    as *const _,
                timeout_handler: windows_exception_handler::inproc_timeout_handler::<E, EM, OF, Z>
                    as *const c_void,
            })
        }
    }

    /// Replace the handlers with `nop` handlers, deactivating the handlers
    #[must_use]
    pub fn nop() -> Self {
        let ret;
        #[cfg(any(unix, feature = "std"))]
        {
            ret = Self {
                crash_handler: ptr::null(),
                timeout_handler: ptr::null(),
            };
        }
        #[cfg(not(any(unix, feature = "std")))]
        {
            ret = Self {};
        }
        ret
    }
}

/// The global state of the in-process harness.
#[derive(Debug)]
pub struct InProcessExecutorHandlerData {
    state_ptr: *mut c_void,
    event_mgr_ptr: *mut c_void,
    fuzzer_ptr: *mut c_void,
    executor_ptr: *const c_void,
    pub(crate) current_input_ptr: *const c_void,
    pub(crate) in_handler: bool,

    /// The timeout handler
    #[cfg(any(unix, feature = "std"))]
    crash_handler: *const c_void,
    /// The timeout handler
    #[cfg(any(unix, feature = "std"))]
    timeout_handler: *const c_void,

    #[cfg(all(windows, feature = "std"))]
    pub(crate) ptp_timer: Option<PTP_TIMER>,
    #[cfg(all(windows, feature = "std"))]
    pub(crate) in_target: u64,
    #[cfg(all(windows, feature = "std"))]
    pub(crate) critical: *mut c_void,
    #[cfg(all(windows, feature = "std"))]
    pub(crate) timeout_input_ptr: *mut c_void,

    #[cfg(any(unix, feature = "std"))]
    pub(crate) timeout_executor_ptr: *mut c_void,
}

unsafe impl Send for InProcessExecutorHandlerData {}
unsafe impl Sync for InProcessExecutorHandlerData {}

impl InProcessExecutorHandlerData {
    #[cfg(any(unix, feature = "std"))]
    fn executor_mut<'a, E>(&self) -> &'a mut E {
        unsafe { (self.executor_ptr as *mut E).as_mut().unwrap() }
    }

    #[cfg(any(unix, feature = "std"))]
    fn state_mut<'a, S>(&self) -> &'a mut S {
        unsafe { (self.state_ptr as *mut S).as_mut().unwrap() }
    }

    #[cfg(any(unix, feature = "std"))]
    fn event_mgr_mut<'a, EM>(&self) -> &'a mut EM {
        unsafe { (self.event_mgr_ptr as *mut EM).as_mut().unwrap() }
    }

    #[cfg(any(unix, feature = "std"))]
    fn fuzzer_mut<'a, Z>(&self) -> &'a mut Z {
        unsafe { (self.fuzzer_ptr as *mut Z).as_mut().unwrap() }
    }

    #[cfg(any(unix, feature = "std"))]
    fn take_current_input<'a, I>(&mut self) -> &'a I {
        let r = unsafe { (self.current_input_ptr as *const I).as_ref().unwrap() };
        self.current_input_ptr = ptr::null();
        r
    }

    #[cfg(any(unix, feature = "std"))]
    pub(crate) fn is_valid(&self) -> bool {
        !self.current_input_ptr.is_null()
    }

    #[cfg(any(unix, feature = "std"))]
    fn timeout_executor_mut<'a, E>(&self) -> &'a mut crate::executors::timeout::TimeoutExecutor<E> {
        unsafe {
            (self.timeout_executor_ptr as *mut crate::executors::timeout::TimeoutExecutor<E>)
                .as_mut()
                .unwrap()
        }
    }

    #[cfg(any(unix, feature = "std"))]
    fn set_in_handler(&mut self, v: bool) -> bool {
        let old = self.in_handler;
        self.in_handler = v;
        old
    }
}

/// Exception handling needs some nasty unsafe.
pub(crate) static mut GLOBAL_STATE: InProcessExecutorHandlerData = InProcessExecutorHandlerData {
    // The state ptr for signal handling
    state_ptr: null_mut(),
    // The event manager ptr for signal handling
    event_mgr_ptr: null_mut(),
    // The fuzzer ptr for signal handling
    fuzzer_ptr: null_mut(),
    // The executor ptr for signal handling
    executor_ptr: ptr::null(),
    // The current input for signal handling
    current_input_ptr: ptr::null(),

    in_handler: false,

    // The crash handler fn
    #[cfg(any(unix, feature = "std"))]
    crash_handler: ptr::null(),
    // The timeout handler fn
    #[cfg(any(unix, feature = "std"))]
    timeout_handler: ptr::null(),
    #[cfg(all(windows, feature = "std"))]
    ptp_timer: None,
    #[cfg(all(windows, feature = "std"))]
    in_target: 0,
    #[cfg(all(windows, feature = "std"))]
    critical: null_mut(),
    #[cfg(all(windows, feature = "std"))]
    timeout_input_ptr: null_mut(),

    #[cfg(any(unix, feature = "std"))]
    timeout_executor_ptr: null_mut(),
};

/// Get the inprocess [`crate::state::State`]
#[must_use]
pub fn inprocess_get_state<'a, S>() -> Option<&'a mut S> {
    unsafe { (GLOBAL_STATE.state_ptr as *mut S).as_mut() }
}

/// Get the [`crate::events::EventManager`]
#[must_use]
pub fn inprocess_get_event_manager<'a, EM>() -> Option<&'a mut EM> {
    unsafe { (GLOBAL_STATE.event_mgr_ptr as *mut EM).as_mut() }
}

/// Gets the inprocess [`crate::fuzzer::Fuzzer`]
#[must_use]
pub fn inprocess_get_fuzzer<'a, F>() -> Option<&'a mut F> {
    unsafe { (GLOBAL_STATE.fuzzer_ptr as *mut F).as_mut() }
}

/// Gets the inprocess [`Executor`]
#[must_use]
pub fn inprocess_get_executor<'a, E>() -> Option<&'a mut E> {
    unsafe { (GLOBAL_STATE.executor_ptr as *mut E).as_mut() }
}

/// Gets the inprocess input
#[must_use]
pub fn inprocess_get_input<'a, I>() -> Option<&'a I> {
    unsafe { (GLOBAL_STATE.current_input_ptr as *const I).as_ref() }
}

/// Know if we ar eexecuting in a crash/timeout handler
#[must_use]
pub fn inprocess_in_handler() -> bool {
    unsafe { GLOBAL_STATE.in_handler }
}

use crate::{
    corpus::{Corpus, Testcase},
    events::Event,
    state::HasMetadata,
};

#[inline]
#[allow(clippy::too_many_arguments)]
/// Save state if it is an objective
pub fn run_observers_and_save_state<E, EM, OF, Z>(
    executor: &mut E,
    state: &mut E::State,
    input: &<E::State as UsesInput>::Input,
    fuzzer: &mut Z,
    event_mgr: &mut EM,
    exitkind: ExitKind,
) where
    E: HasObservers,
    EM: EventFirer<State = E::State> + EventRestarter<State = E::State>,
    OF: Feedback<E::State>,
    E::State: HasSolutions + HasClientPerfMonitor + HasCorpus,
    Z: HasObjective<Objective = OF, State = E::State>,
{
    let observers = executor.observers_mut();

    observers
        .post_exec_all(state, input, &exitkind)
        .expect("Observers post_exec_all failed");

    let interesting = fuzzer
        .objective_mut()
        .is_interesting(state, event_mgr, input, observers, &exitkind)
        .expect("In run_observers_and_save_state objective failure.");

    if interesting {
        let mut new_testcase = Testcase::new(input.clone());
        new_testcase.add_metadata(exitkind);
        new_testcase.set_parent_id_optional(*state.corpus().current());
        fuzzer
            .objective_mut()
            .append_metadata(state, observers, &mut new_testcase)
            .expect("Failed adding metadata");
        state
            .solutions_mut()
            .add(new_testcase)
            .expect("In run_observers_and_save_state solutions failure.");
        event_mgr
            .fire(
                state,
                Event::Objective {
                    objective_size: state.solutions().count(),
                },
            )
            .expect("Could not save state in run_observers_and_save_state");
    }

    // Serialize the state and wait safely for the broker to read pending messages
    event_mgr.on_restart(state).unwrap();

    log::info!("Bye!");
}

/// The inprocess executor singal handling code for unix
#[cfg(unix)]
pub mod unix_signal_handler {
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use alloc::{boxed::Box, string::String};
    use core::mem::transmute;
    #[cfg(feature = "std")]
    use std::{io::Write, panic};

    use libafl_bolts::os::unix_signals::{ucontext_t, Handler, Signal};
    use libc::siginfo_t;

    #[cfg(feature = "std")]
    use crate::inputs::Input;
    use crate::{
        events::{EventFirer, EventRestarter},
        executors::{
            inprocess::{run_observers_and_save_state, InProcessExecutorHandlerData, GLOBAL_STATE},
            Executor, ExitKind, HasObservers,
        },
        feedbacks::Feedback,
        fuzzer::HasObjective,
        inputs::UsesInput,
        state::{HasClientPerfMonitor, HasCorpus, HasSolutions},
    };

    pub(crate) type HandlerFuncPtr = unsafe fn(
        Signal,
        &mut siginfo_t,
        Option<&mut ucontext_t>,
        data: &mut InProcessExecutorHandlerData,
    );

    /// A handler that does nothing.
    /*pub fn nop_handler(
        _signal: Signal,
        _info: &mut siginfo_t,
        _context: Option<&mut ucontext_t>,
        _data: &mut InProcessExecutorHandlerData,
    ) {
    }*/

    #[cfg(unix)]
    impl Handler for InProcessExecutorHandlerData {
        fn handle(
            &mut self,
            signal: Signal,
            info: &mut siginfo_t,
            context: Option<&mut ucontext_t>,
        ) {
            unsafe {
                let data = &mut GLOBAL_STATE;
                let in_handler = data.set_in_handler(true);
                match signal {
                    Signal::SigUser2 | Signal::SigAlarm => {
                        if !data.timeout_handler.is_null() {
                            let func: HandlerFuncPtr = transmute(data.timeout_handler);
                            (func)(signal, info, context, data);
                        }
                    }
                    _ => {
                        if !data.crash_handler.is_null() {
                            let func: HandlerFuncPtr = transmute(data.crash_handler);
                            (func)(signal, info, context, data);
                        }
                    }
                }
                data.set_in_handler(in_handler);
            }
        }

        fn signals(&self) -> Vec<Signal> {
            vec![
                Signal::SigAlarm,
                Signal::SigUser2,
                Signal::SigAbort,
                Signal::SigBus,
                Signal::SigPipe,
                Signal::SigFloatingPointException,
                Signal::SigIllegalInstruction,
                Signal::SigSegmentationFault,
                Signal::SigTrap,
            ]
        }
    }

    /// invokes the `post_exec` hook on all observer in case of panic
    #[cfg(feature = "std")]
    pub fn setup_panic_hook<E, EM, OF, Z>()
    where
        E: HasObservers,
        EM: EventFirer<State = E::State> + EventRestarter<State = E::State>,
        OF: Feedback<E::State>,
        E::State: HasSolutions + HasClientPerfMonitor + HasCorpus,
        Z: HasObjective<Objective = OF, State = E::State>,
    {
        let old_hook = panic::take_hook();
        panic::set_hook(Box::new(move |panic_info| {
            old_hook(panic_info);
            let data = unsafe { &mut GLOBAL_STATE };
            let in_handler = data.set_in_handler(true);
            if data.is_valid() {
                // We are fuzzing!
                let executor = data.executor_mut::<E>();
                let state = data.state_mut::<E::State>();
                let input = data.take_current_input::<<E::State as UsesInput>::Input>();
                let fuzzer = data.fuzzer_mut::<Z>();
                let event_mgr = data.event_mgr_mut::<EM>();

                run_observers_and_save_state::<E, EM, OF, Z>(
                    executor,
                    state,
                    input,
                    fuzzer,
                    event_mgr,
                    ExitKind::Crash,
                );

                unsafe {
                    libc::_exit(128 + 6);
                } // SIGABRT exit code
            }
            data.set_in_handler(in_handler);
        }));
    }

    /// Timeout-Handler for in-process fuzzing.
    /// It will store the current State to shmem, then exit.
    ///
    /// # Safety
    /// Well, signal handling is not safe
    #[cfg(unix)]
    pub unsafe fn inproc_timeout_handler<E, EM, OF, Z>(
        _signal: Signal,
        _info: &mut siginfo_t,
        _context: Option<&mut ucontext_t>,
        data: &mut InProcessExecutorHandlerData,
    ) where
        E: HasObservers,
        EM: EventFirer<State = E::State> + EventRestarter<State = E::State>,
        OF: Feedback<E::State>,
        E::State: HasSolutions + HasClientPerfMonitor + HasCorpus,
        Z: HasObjective<Objective = OF, State = E::State>,
    {
        if !data.timeout_executor_ptr.is_null()
            && data.timeout_executor_mut::<E>().handle_timeout(data)
        {
            return;
        }

        if !data.is_valid() {
            log::warn!("TIMEOUT or SIGUSR2 happened, but currently not fuzzing.");
            return;
        }

        let executor = data.executor_mut::<E>();
        let state = data.state_mut::<E::State>();
        let event_mgr = data.event_mgr_mut::<EM>();
        let fuzzer = data.fuzzer_mut::<Z>();
        let input = data.take_current_input::<<E::State as UsesInput>::Input>();

        log::error!("Timeout in fuzz run.");

        run_observers_and_save_state::<E, EM, OF, Z>(
            executor,
            state,
            input,
            fuzzer,
            event_mgr,
            ExitKind::Timeout,
        );

        libc::_exit(55);
    }

    /// Crash-Handler for in-process fuzzing.
    /// Will be used for signal handling.
    /// It will store the current State to shmem, then exit.
    ///
    /// # Safety
    /// Well, signal handling is not safe
    #[allow(clippy::too_many_lines)]
    pub unsafe fn inproc_crash_handler<E, EM, OF, Z>(
        signal: Signal,
        _info: &mut siginfo_t,
        _context: Option<&mut ucontext_t>,
        data: &mut InProcessExecutorHandlerData,
    ) where
        E: Executor<EM, Z> + HasObservers,
        EM: EventFirer<State = E::State> + EventRestarter<State = E::State>,
        OF: Feedback<E::State>,
        E::State: HasSolutions + HasClientPerfMonitor + HasCorpus,
        Z: HasObjective<Objective = OF, State = E::State>,
    {
        #[cfg(all(target_os = "android", target_arch = "aarch64"))]
        let _context = &mut *(((_context as *mut _ as *mut libc::c_void as usize) + 128)
            as *mut libc::c_void as *mut ucontext_t);

        log::error!("Crashed with {signal}");
        if data.is_valid() {
            let executor = data.executor_mut::<E>();
            // disarms timeout in case of TimeoutExecutor
            executor.post_run_reset();
            let state = data.state_mut::<E::State>();
            let event_mgr = data.event_mgr_mut::<EM>();
            let fuzzer = data.fuzzer_mut::<Z>();
            let input = data.take_current_input::<<E::State as UsesInput>::Input>();

            log::error!("Child crashed!");

            #[cfg(all(feature = "std", unix))]
            {
                let mut bsod = Vec::new();
                {
                    let mut writer = std::io::BufWriter::new(&mut bsod);
                    writeln!(writer, "input: {:?}", input.generate_name(0)).unwrap();
                    libafl_bolts::minibsod::generate_minibsod(
                        &mut writer,
                        signal,
                        _info,
                        _context.as_deref(),
                    )
                    .unwrap();
                    writer.flush().unwrap();
                }
                log::error!("{}", std::str::from_utf8(&bsod).unwrap());
            }

            run_observers_and_save_state::<E, EM, OF, Z>(
                executor,
                state,
                input,
                fuzzer,
                event_mgr,
                ExitKind::Crash,
            );
        } else {
            {
                log::error!("Double crash\n");
                #[cfg(target_os = "android")]
                let si_addr = (_info._pad[0] as i64) | ((_info._pad[1] as i64) << 32);
                #[cfg(not(target_os = "android"))]
                let si_addr = { _info.si_addr() as usize };

                log::error!(
                    "We crashed at addr 0x{si_addr:x}, but are not in the target... Bug in the fuzzer? Exiting."
                );

                #[cfg(all(feature = "std", unix))]
                {
                    let mut bsod = Vec::new();
                    {
                        let mut writer = std::io::BufWriter::new(&mut bsod);
                        libafl_bolts::minibsod::generate_minibsod(
                            &mut writer,
                            signal,
                            _info,
                            _context.as_deref(),
                        )
                        .unwrap();
                        writer.flush().unwrap();
                    }
                    log::error!("{}", std::str::from_utf8(&bsod).unwrap());
                }
            }

            #[cfg(feature = "std")]
            {
                log::error!("Type QUIT to restart the child");
                let mut line = String::new();
                while line.trim() != "QUIT" {
                    std::io::stdin().read_line(&mut line).unwrap();
                }
            }

            // TODO tell the parent to not restart
        }

        libc::_exit(128 + (signal as i32));
    }
}

/// Same as `inproc_crash_handler`, but this is called when address sanitizer exits, not from the exception handler
#[cfg(all(windows, feature = "std"))]
pub mod windows_asan_handler {
    use alloc::string::String;
    use core::sync::atomic::{compiler_fence, Ordering};

    use windows::Win32::System::Threading::{
        EnterCriticalSection, LeaveCriticalSection, CRITICAL_SECTION,
    };

    use crate::{
        events::{EventFirer, EventRestarter},
        executors::{
            inprocess::{run_observers_and_save_state, GLOBAL_STATE},
            Executor, ExitKind, HasObservers,
        },
        feedbacks::Feedback,
        fuzzer::HasObjective,
        inputs::UsesInput,
        state::{HasClientPerfMonitor, HasCorpus, HasSolutions},
    };

    /// # Safety
    /// ASAN deatch handler
    pub unsafe extern "C" fn asan_death_handler<E, EM, OF, Z>()
    where
        E: Executor<EM, Z> + HasObservers,
        EM: EventFirer<State = E::State> + EventRestarter<State = E::State>,
        OF: Feedback<E::State>,
        E::State: HasSolutions + HasClientPerfMonitor + HasCorpus,
        Z: HasObjective<Objective = OF, State = E::State>,
    {
        let data = &mut GLOBAL_STATE;
        data.set_in_handler(true);
        // Have we set a timer_before?
        if data.ptp_timer.is_some() {
            /*
                We want to prevent the timeout handler being run while the main thread is executing the crash handler
                Timeout handler runs if it has access to the critical section or data.in_target == 0
                Writing 0 to the data.in_target makes the timeout handler makes the timeout handler invalid.
            */
            compiler_fence(Ordering::SeqCst);
            EnterCriticalSection(data.critical as *mut CRITICAL_SECTION);
            compiler_fence(Ordering::SeqCst);
            data.in_target = 0;
            compiler_fence(Ordering::SeqCst);
            LeaveCriticalSection(data.critical as *mut CRITICAL_SECTION);
            compiler_fence(Ordering::SeqCst);
        }

        log::error!("ASAN detected crash!");
        if data.current_input_ptr.is_null() {
            {
                log::error!("Double crash\n");
                log::error!(
                "ASAN detected crash but we're not in the target... Bug in the fuzzer? Exiting.",
                );
            }
            #[cfg(feature = "std")]
            {
                log::error!("Type QUIT to restart the child");
                let mut line = String::new();
                while line.trim() != "QUIT" {
                    std::io::stdin().read_line(&mut line).unwrap();
                }
            }

            // TODO tell the parent to not restart
        } else {
            let executor = data.executor_mut::<E>();
            // reset timer
            if data.ptp_timer.is_some() {
                executor.post_run_reset();
                data.ptp_timer = None;
            }

            let state = data.state_mut::<E::State>();
            let fuzzer = data.fuzzer_mut::<Z>();
            let event_mgr = data.event_mgr_mut::<EM>();

            log::error!("Child crashed!");

            // Make sure we don't crash in the crash handler forever.
            let input = data.take_current_input::<<E::State as UsesInput>::Input>();

            run_observers_and_save_state::<E, EM, OF, Z>(
                executor,
                state,
                input,
                fuzzer,
                event_mgr,
                ExitKind::Crash,
            );
        }
        // Don't need to exit, Asan will exit for us
        // ExitProcess(1);
    }
}

#[cfg(all(windows, feature = "std"))]
pub mod windows_exception_handler {
    #[cfg(feature = "std")]
    use alloc::boxed::Box;
    use alloc::{string::String, vec::Vec};
    use core::{
        ffi::c_void,
        mem::transmute,
        ptr,
        sync::atomic::{compiler_fence, Ordering},
    };
    #[cfg(feature = "std")]
    use std::panic;

    use libafl_bolts::os::windows_exceptions::{
        ExceptionCode, Handler, CRASH_EXCEPTIONS, EXCEPTION_HANDLERS_SIZE, EXCEPTION_POINTERS,
    };
    use windows::Win32::System::Threading::{
        EnterCriticalSection, ExitProcess, LeaveCriticalSection, CRITICAL_SECTION,
    };

    use crate::{
        events::{EventFirer, EventRestarter},
        executors::{
            inprocess::{
                run_observers_and_save_state, HasInProcessHandlers, InProcessExecutorHandlerData,
                GLOBAL_STATE,
            },
            Executor, ExitKind, HasObservers,
        },
        feedbacks::Feedback,
        fuzzer::HasObjective,
        inputs::UsesInput,
        state::{HasClientPerfMonitor, HasCorpus, HasSolutions},
    };

    pub(crate) type HandlerFuncPtr =
        unsafe fn(*mut EXCEPTION_POINTERS, &mut InProcessExecutorHandlerData);

    /*pub unsafe fn nop_handler(
        _code: ExceptionCode,
        _exception_pointers: *mut EXCEPTION_POINTERS,
        _data: &mut InProcessExecutorHandlerData,
    ) {
    }*/

    impl Handler for InProcessExecutorHandlerData {
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn handle(&mut self, _code: ExceptionCode, exception_pointers: *mut EXCEPTION_POINTERS) {
            unsafe {
                let data = &mut GLOBAL_STATE;
                let in_handler = data.set_in_handler(true);
                if !data.crash_handler.is_null() {
                    let func: HandlerFuncPtr = transmute(data.crash_handler);
                    (func)(exception_pointers, data);
                }
                data.set_in_handler(in_handler);
            }
        }

        fn exceptions(&self) -> Vec<ExceptionCode> {
            let crash_list = CRASH_EXCEPTIONS.to_vec();
            assert!(crash_list.len() < EXCEPTION_HANDLERS_SIZE - 1);
            crash_list
        }
    }

    /// invokes the `post_exec` hook on all observer in case of panic
    ///
    /// # Safety
    /// Well, exception handling is not safe
    #[cfg(feature = "std")]
    pub fn setup_panic_hook<E, EM, OF, Z>()
    where
        E: HasObservers,
        EM: EventFirer<State = E::State> + EventRestarter<State = E::State>,
        OF: Feedback<E::State>,
        E::State: HasSolutions + HasClientPerfMonitor + HasCorpus,
        Z: HasObjective<Objective = OF, State = E::State>,
    {
        let old_hook = panic::take_hook();
        panic::set_hook(Box::new(move |panic_info| {
            let data = unsafe { &mut GLOBAL_STATE };
            let in_handler = data.set_in_handler(true);
            // Have we set a timer_before?
            unsafe {
                if data.ptp_timer.is_some() {
                    /*
                        We want to prevent the timeout handler being run while the main thread is executing the crash handler
                        Timeout handler runs if it has access to the critical section or data.in_target == 0
                        Writing 0 to the data.in_target makes the timeout handler makes the timeout handler invalid.
                    */
                    compiler_fence(Ordering::SeqCst);
                    EnterCriticalSection(data.critical as *mut CRITICAL_SECTION);
                    compiler_fence(Ordering::SeqCst);
                    data.in_target = 0;
                    compiler_fence(Ordering::SeqCst);
                    LeaveCriticalSection(data.critical as *mut CRITICAL_SECTION);
                    compiler_fence(Ordering::SeqCst);
                }
            }

            if data.is_valid() {
                // We are fuzzing!
                let executor = data.executor_mut::<E>();
                let state = data.state_mut::<E::State>();
                let fuzzer = data.fuzzer_mut::<Z>();
                let event_mgr = data.event_mgr_mut::<EM>();

                let input = data.take_current_input::<<E::State as UsesInput>::Input>();

                run_observers_and_save_state::<E, EM, OF, Z>(
                    executor,
                    state,
                    input,
                    fuzzer,
                    event_mgr,
                    ExitKind::Crash,
                );

                unsafe {
                    ExitProcess(1);
                }
            }
            old_hook(panic_info);
            data.set_in_handler(in_handler);
        }));
    }

    /// Timeout handler for windows
    ///
    /// # Safety
    /// Well, exception handling is not safe
    pub unsafe extern "system" fn inproc_timeout_handler<E, EM, OF, Z>(
        _p0: *mut u8,
        global_state: *mut c_void,
        _p1: *mut u8,
    ) where
        E: HasObservers + HasInProcessHandlers,
        EM: EventFirer<State = E::State> + EventRestarter<State = E::State>,
        OF: Feedback<E::State>,
        E::State: HasSolutions + HasClientPerfMonitor + HasCorpus,
        Z: HasObjective<Objective = OF, State = E::State>,
    {
        let data: &mut InProcessExecutorHandlerData =
            &mut *(global_state as *mut InProcessExecutorHandlerData);
        compiler_fence(Ordering::SeqCst);
        EnterCriticalSection((data.critical as *mut CRITICAL_SECTION).as_mut().unwrap());
        compiler_fence(Ordering::SeqCst);

        if !data.timeout_executor_ptr.is_null()
            && data.timeout_executor_mut::<E>().handle_timeout(data)
        {
            compiler_fence(Ordering::SeqCst);
            LeaveCriticalSection((data.critical as *mut CRITICAL_SECTION).as_mut().unwrap());
            compiler_fence(Ordering::SeqCst);

            return;
        }

        if data.in_target == 1 {
            let executor = data.executor_mut::<E>();
            let state = data.state_mut::<E::State>();
            let fuzzer = data.fuzzer_mut::<Z>();
            let event_mgr = data.event_mgr_mut::<EM>();

            if data.timeout_input_ptr.is_null() {
                log::error!("TIMEOUT or SIGUSR2 happened, but currently not fuzzing. Exiting");
            } else {
                log::error!("Timeout in fuzz run.");

                let input = (data.timeout_input_ptr as *const <E::State as UsesInput>::Input)
                    .as_ref()
                    .unwrap();
                data.timeout_input_ptr = ptr::null_mut();

                run_observers_and_save_state::<E, EM, OF, Z>(
                    executor,
                    state,
                    input,
                    fuzzer,
                    event_mgr,
                    ExitKind::Timeout,
                );

                compiler_fence(Ordering::SeqCst);

                ExitProcess(1);
            }
        }
        compiler_fence(Ordering::SeqCst);
        LeaveCriticalSection((data.critical as *mut CRITICAL_SECTION).as_mut().unwrap());
        compiler_fence(Ordering::SeqCst);
        // log::info!("TIMER INVOKED!");
    }

    /// Crash handler for windows
    ///
    /// # Safety
    /// Well, exception handling is not safe
    #[allow(clippy::too_many_lines)]
    pub unsafe fn inproc_crash_handler<E, EM, OF, Z>(
        exception_pointers: *mut EXCEPTION_POINTERS,
        data: &mut InProcessExecutorHandlerData,
    ) where
        E: Executor<EM, Z> + HasObservers,
        EM: EventFirer<State = E::State> + EventRestarter<State = E::State>,
        OF: Feedback<E::State>,
        E::State: HasSolutions + HasClientPerfMonitor + HasCorpus,
        Z: HasObjective<Objective = OF, State = E::State>,
    {
        // Have we set a timer_before?
        if data.ptp_timer.is_some() {
            /*
                We want to prevent the timeout handler being run while the main thread is executing the crash handler
                Timeout handler runs if it has access to the critical section or data.in_target == 0
                Writing 0 to the data.in_target makes the timeout handler makes the timeout handler invalid.
            */
            compiler_fence(Ordering::SeqCst);
            EnterCriticalSection(data.critical as *mut CRITICAL_SECTION);
            compiler_fence(Ordering::SeqCst);
            data.in_target = 0;
            compiler_fence(Ordering::SeqCst);
            LeaveCriticalSection(data.critical as *mut CRITICAL_SECTION);
            compiler_fence(Ordering::SeqCst);
        }

        // Is this really crash?
        let mut is_crash = true;
        #[cfg(feature = "std")]
        if let Some(exception_pointers) = exception_pointers.as_mut() {
            let code = ExceptionCode::try_from(
                exception_pointers
                    .ExceptionRecord
                    .as_mut()
                    .unwrap()
                    .ExceptionCode
                    .0,
            )
            .unwrap();

            let exception_list = data.exceptions();
            if exception_list.contains(&code) {
                log::error!("Crashed with {code}");
            } else {
                // log::trace!("Exception code received, but {code} is not in CRASH_EXCEPTIONS");
                is_crash = false;
            }
        } else {
            log::error!("Crashed without exception (probably due to SIGABRT)");
        };

        if data.current_input_ptr.is_null() {
            {
                log::error!("Double crash\n");
                let crash_addr = exception_pointers
                    .as_mut()
                    .unwrap()
                    .ExceptionRecord
                    .as_mut()
                    .unwrap()
                    .ExceptionAddress as usize;

                log::error!(
                "We crashed at addr 0x{crash_addr:x}, but are not in the target... Bug in the fuzzer? Exiting."
                );
            }
            #[cfg(feature = "std")]
            {
                log::error!("Type QUIT to restart the child");
                let mut line = String::new();
                while line.trim() != "QUIT" {
                    std::io::stdin().read_line(&mut line).unwrap();
                }
            }

            // TODO tell the parent to not restart
        } else {
            let executor = data.executor_mut::<E>();
            // reset timer
            if data.ptp_timer.is_some() {
                executor.post_run_reset();
                data.ptp_timer = None;
            }

            let state = data.state_mut::<E::State>();
            let fuzzer = data.fuzzer_mut::<Z>();
            let event_mgr = data.event_mgr_mut::<EM>();

            if is_crash {
                log::error!("Child crashed!");
            } else {
                // log::info!("Exception received!");
            }

            // Make sure we don't crash in the crash handler forever.
            if is_crash {
                let input = data.take_current_input::<<E::State as UsesInput>::Input>();

                run_observers_and_save_state::<E, EM, OF, Z>(
                    executor,
                    state,
                    input,
                    fuzzer,
                    event_mgr,
                    ExitKind::Crash,
                );
            } else {
                // This is not worth saving
            }
        }

        if is_crash {
            log::info!("Exiting!");
            ExitProcess(1);
        }
        // log::info!("Not Exiting!");
    }
}

/// The signature of the crash handler function
#[cfg(all(feature = "std", unix))]
pub(crate) type ForkHandlerFuncPtr = unsafe fn(
    Signal,
    &mut siginfo_t,
    Option<&mut ucontext_t>,
    data: &mut InProcessForkExecutorGlobalData,
);

/// The inmem fork executor's handlers.
#[cfg(all(feature = "std", unix))]
#[derive(Debug)]
pub struct InChildProcessHandlers {
    /// On crash C function pointer
    pub crash_handler: *const c_void,
    /// On timeout C function pointer
    pub timeout_handler: *const c_void,
}

#[cfg(all(feature = "std", unix))]
impl InChildProcessHandlers {
    /// Call before running a target.
    pub fn pre_run_target<E, I, S>(&self, executor: &E, state: &mut S, input: &I) {
        unsafe {
            let data = &mut FORK_EXECUTOR_GLOBAL_DATA;
            write_volatile(
                &mut data.executor_ptr,
                executor as *const _ as *const c_void,
            );
            write_volatile(
                &mut data.current_input_ptr,
                input as *const _ as *const c_void,
            );
            write_volatile(&mut data.state_ptr, state as *mut _ as *mut c_void);
            data.crash_handler = self.crash_handler;
            data.timeout_handler = self.timeout_handler;
            compiler_fence(Ordering::SeqCst);
        }
    }

    /// Create new [`InChildProcessHandlers`].
    pub fn new<E>() -> Result<Self, Error>
    where
        E: HasObservers,
    {
        #[cfg_attr(miri, allow(unused_variables))]
        unsafe {
            let data = &mut FORK_EXECUTOR_GLOBAL_DATA;
            // child_signal_handlers::setup_child_panic_hook::<E, I, OT, S>();
            #[cfg(not(miri))]
            setup_signal_handler(data)?;
            compiler_fence(Ordering::SeqCst);
            Ok(Self {
                crash_handler: child_signal_handlers::child_crash_handler::<E> as *const c_void,
                timeout_handler: ptr::null(),
            })
        }
    }

    /// Create new [`InChildProcessHandlers`].
    pub fn with_timeout<E>() -> Result<Self, Error>
    where
        E: HasObservers,
    {
        #[cfg_attr(miri, allow(unused_variables))]
        unsafe {
            let data = &mut FORK_EXECUTOR_GLOBAL_DATA;
            // child_signal_handlers::setup_child_panic_hook::<E, I, OT, S>();
            #[cfg(not(miri))]
            setup_signal_handler(data)?;
            compiler_fence(Ordering::SeqCst);
            Ok(Self {
                crash_handler: child_signal_handlers::child_crash_handler::<E> as *const c_void,
                timeout_handler: child_signal_handlers::child_timeout_handler::<E> as *const c_void,
            })
        }
    }

    /// Replace the handlers with `nop` handlers, deactivating the handlers
    #[must_use]
    pub fn nop() -> Self {
        Self {
            crash_handler: ptr::null(),
            timeout_handler: ptr::null(),
        }
    }
}

/// The global state of the in-process-fork harness.
#[cfg(all(feature = "std", unix))]
#[derive(Debug)]
pub(crate) struct InProcessForkExecutorGlobalData {
    /// Stores a pointer to the fork executor struct
    pub executor_ptr: *const c_void,
    /// Stores a pointer to the state
    pub state_ptr: *const c_void,
    /// Stores a pointer to the current input
    pub current_input_ptr: *const c_void,
    /// Stores a pointer to the crash_handler function
    pub crash_handler: *const c_void,
    /// Stores a pointer to the timeout_handler function
    pub timeout_handler: *const c_void,
}

#[cfg(all(feature = "std", unix))]
unsafe impl Sync for InProcessForkExecutorGlobalData {}
#[cfg(all(feature = "std", unix))]
unsafe impl Send for InProcessForkExecutorGlobalData {}

#[cfg(all(feature = "std", unix))]
impl InProcessForkExecutorGlobalData {
    fn executor_mut<'a, E>(&self) -> &'a mut E {
        unsafe { (self.executor_ptr as *mut E).as_mut().unwrap() }
    }

    fn state_mut<'a, S>(&self) -> &'a mut S {
        unsafe { (self.state_ptr as *mut S).as_mut().unwrap() }
    }

    /*fn current_input<'a, I>(&self) -> &'a I {
        unsafe { (self.current_input_ptr as *const I).as_ref().unwrap() }
    }*/

    fn take_current_input<'a, I>(&mut self) -> &'a I {
        let r = unsafe { (self.current_input_ptr as *const I).as_ref().unwrap() };
        self.current_input_ptr = ptr::null();
        r
    }

    fn is_valid(&self) -> bool {
        !self.current_input_ptr.is_null()
    }
}

/// a static variable storing the global state
#[cfg(all(feature = "std", unix))]
pub(crate) static mut FORK_EXECUTOR_GLOBAL_DATA: InProcessForkExecutorGlobalData =
    InProcessForkExecutorGlobalData {
        executor_ptr: ptr::null(),
        state_ptr: ptr::null(),
        current_input_ptr: ptr::null(),
        crash_handler: ptr::null(),
        timeout_handler: ptr::null(),
    };

#[cfg(all(feature = "std", unix))]
impl Handler for InProcessForkExecutorGlobalData {
    fn handle(&mut self, signal: Signal, info: &mut siginfo_t, context: Option<&mut ucontext_t>) {
        match signal {
            Signal::SigUser2 | Signal::SigAlarm => unsafe {
                if !FORK_EXECUTOR_GLOBAL_DATA.timeout_handler.is_null() {
                    let func: ForkHandlerFuncPtr =
                        transmute(FORK_EXECUTOR_GLOBAL_DATA.timeout_handler);
                    (func)(signal, info, context, &mut FORK_EXECUTOR_GLOBAL_DATA);
                }
            },
            _ => unsafe {
                if !FORK_EXECUTOR_GLOBAL_DATA.crash_handler.is_null() {
                    let func: ForkHandlerFuncPtr =
                        transmute(FORK_EXECUTOR_GLOBAL_DATA.crash_handler);
                    (func)(signal, info, context, &mut FORK_EXECUTOR_GLOBAL_DATA);
                }
            },
        }
    }

    fn signals(&self) -> Vec<Signal> {
        vec![
            Signal::SigAlarm,
            Signal::SigUser2,
            Signal::SigAbort,
            Signal::SigBus,
            Signal::SigPipe,
            Signal::SigFloatingPointException,
            Signal::SigIllegalInstruction,
            Signal::SigSegmentationFault,
            Signal::SigTrap,
        ]
    }
}

#[repr(C)]
#[cfg(all(feature = "std", unix, not(target_os = "linux")))]
struct Timeval {
    pub tv_sec: i64,
    pub tv_usec: i64,
}

#[cfg(all(feature = "std", unix, not(target_os = "linux")))]
impl Debug for Timeval {
    #[allow(clippy::cast_sign_loss)]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Timeval {{ tv_sec: {:?}, tv_usec: {:?} (tv: {:?}) }}",
            self.tv_sec,
            self.tv_usec,
            Duration::new(self.tv_sec as _, (self.tv_usec * 1000) as _)
        )
    }
}

#[repr(C)]
#[cfg(all(feature = "std", unix, not(target_os = "linux")))]
#[derive(Debug)]
struct Itimerval {
    pub it_interval: Timeval,
    pub it_value: Timeval,
}

#[cfg(all(feature = "std", unix, not(target_os = "linux")))]
extern "C" {
    fn setitimer(
        which: libc::c_int,
        new_value: *mut Itimerval,
        old_value: *mut Itimerval,
    ) -> libc::c_int;
}

#[cfg(all(feature = "std", unix, not(target_os = "linux")))]
const ITIMER_REAL: libc::c_int = 0;

/// [`InProcessForkExecutor`] is an executor that forks the current process before each execution.
#[cfg(all(feature = "std", unix))]
pub struct InProcessForkExecutor<'a, H, OT, S, SP>
where
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    OT: ObserversTuple<S>,
    S: UsesInput,
    SP: ShMemProvider,
{
    harness_fn: &'a mut H,
    shmem_provider: SP,
    observers: OT,
    handlers: InChildProcessHandlers,
    phantom: PhantomData<S>,
}

/// Timeout executor for [`InProcessForkExecutor`]
#[cfg(all(feature = "std", unix))]
pub struct TimeoutInProcessForkExecutor<'a, H, OT, S, SP>
where
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    OT: ObserversTuple<S>,
    S: UsesInput,
    SP: ShMemProvider,
{
    harness_fn: &'a mut H,
    shmem_provider: SP,
    observers: OT,
    handlers: InChildProcessHandlers,
    #[cfg(target_os = "linux")]
    itimerspec: libc::itimerspec,
    #[cfg(all(unix, not(target_os = "linux")))]
    itimerval: Itimerval,
    phantom: PhantomData<S>,
}

#[cfg(all(feature = "std", unix))]
impl<'a, H, OT, S, SP> Debug for InProcessForkExecutor<'a, H, OT, S, SP>
where
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    OT: ObserversTuple<S>,
    S: UsesInput,
    SP: ShMemProvider,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("InProcessForkExecutor")
            .field("observers", &self.observers)
            .field("shmem_provider", &self.shmem_provider)
            .finish()
    }
}

#[cfg(all(feature = "std", unix))]
impl<'a, H, OT, S, SP> Debug for TimeoutInProcessForkExecutor<'a, H, OT, S, SP>
where
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    OT: ObserversTuple<S>,
    S: UsesInput,
    SP: ShMemProvider,
{
    #[cfg(target_os = "linux")]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimeoutInProcessForkExecutor")
            .field("observers", &self.observers)
            .field("shmem_provider", &self.shmem_provider)
            .field("itimerspec", &self.itimerspec)
            .finish()
    }

    #[cfg(not(target_os = "linux"))]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        #[cfg(not(target_os = "linux"))]
        return f
            .debug_struct("TimeoutInProcessForkExecutor")
            .field("observers", &self.observers)
            .field("shmem_provider", &self.shmem_provider)
            .field("itimerval", &self.itimerval)
            .finish();
    }
}

#[cfg(all(feature = "std", unix))]
impl<'a, H, OT, S, SP> UsesState for InProcessForkExecutor<'a, H, OT, S, SP>
where
    H: ?Sized + FnMut(&S::Input) -> ExitKind,
    OT: ObserversTuple<S>,
    S: UsesInput,
    SP: ShMemProvider,
{
    type State = S;
}

#[cfg(all(feature = "std", unix))]
impl<'a, H, OT, S, SP> UsesState for TimeoutInProcessForkExecutor<'a, H, OT, S, SP>
where
    H: ?Sized + FnMut(&S::Input) -> ExitKind,
    OT: ObserversTuple<S>,
    S: UsesInput,
    SP: ShMemProvider,
{
    type State = S;
}

#[cfg(all(feature = "std", unix))]
impl<'a, EM, H, OT, S, SP, Z> Executor<EM, Z> for InProcessForkExecutor<'a, H, OT, S, SP>
where
    EM: UsesState<State = S>,
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    OT: ObserversTuple<S>,
    S: UsesInput + HasExecutions,
    SP: ShMemProvider,
    Z: UsesState<State = S>,
{
    #[allow(unreachable_code)]
    #[inline]
    fn run_target(
        &mut self,
        _fuzzer: &mut Z,
        state: &mut Self::State,
        _mgr: &mut EM,
        input: &Self::Input,
    ) -> Result<ExitKind, Error> {
        *state.executions_mut() += 1;
        unsafe {
            self.shmem_provider.pre_fork()?;
            match fork() {
                Ok(ForkResult::Child) => {
                    // Child
                    self.shmem_provider.post_fork(true)?;

                    self.handlers.pre_run_target(self, state, input);

                    self.observers
                        .pre_exec_child_all(state, input)
                        .expect("Failed to run pre_exec on observers");

                    (self.harness_fn)(input);

                    self.observers
                        .post_exec_child_all(state, input, &ExitKind::Ok)
                        .expect("Failed to run post_exec on observers");

                    libc::_exit(0);

                    Ok(ExitKind::Ok)
                }
                Ok(ForkResult::Parent { child }) => {
                    // Parent
                    // log::info!("from parent {} child is {}", std::process::id(), child);
                    self.shmem_provider.post_fork(false)?;

                    let res = waitpid(child, None)?;

                    match res {
                        WaitStatus::Signaled(_, _, _) => Ok(ExitKind::Crash),
                        WaitStatus::Exited(_, code) => {
                            if code > 128 && code < 160 {
                                // Signal exit codes
                                Ok(ExitKind::Crash)
                            } else {
                                Ok(ExitKind::Ok)
                            }
                        }
                        _ => Ok(ExitKind::Ok),
                    }
                }
                Err(e) => Err(Error::from(e)),
            }
        }
    }
}

#[cfg(all(feature = "std", unix))]
impl<'a, EM, H, OT, S, SP, Z> Executor<EM, Z> for TimeoutInProcessForkExecutor<'a, H, OT, S, SP>
where
    EM: UsesState<State = S>,
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    OT: ObserversTuple<S>,
    S: UsesInput + HasExecutions,
    SP: ShMemProvider,
    Z: UsesState<State = S>,
{
    #[allow(unreachable_code)]
    #[inline]
    fn run_target(
        &mut self,
        _fuzzer: &mut Z,
        state: &mut Self::State,
        _mgr: &mut EM,
        input: &Self::Input,
    ) -> Result<ExitKind, Error> {
        *state.executions_mut() += 1;

        unsafe {
            self.shmem_provider.pre_fork()?;
            match fork() {
                Ok(ForkResult::Child) => {
                    // Child
                    self.shmem_provider.post_fork(true)?;

                    self.handlers.pre_run_target(self, state, input);

                    self.observers
                        .pre_exec_child_all(state, input)
                        .expect("Failed to run post_exec on observers");

                    #[cfg(target_os = "linux")]
                    {
                        let mut timerid: libc::timer_t = null_mut();
                        // creates a new per-process interval timer
                        // we can't do this from the parent, timerid is unique to each process.
                        libc::timer_create(
                            libc::CLOCK_MONOTONIC,
                            null_mut(),
                            addr_of_mut!(timerid),
                        );

                        // log::info!("Set timer! {:#?} {timerid:#?}", self.itimerspec);
                        let _: i32 = libc::timer_settime(
                            timerid,
                            0,
                            addr_of_mut!(self.itimerspec),
                            null_mut(),
                        );
                    }
                    #[cfg(not(target_os = "linux"))]
                    {
                        setitimer(ITIMER_REAL, &mut self.itimerval, null_mut());
                    }
                    // log::trace!("{v:#?} {}", nix::errno::errno());
                    (self.harness_fn)(input);

                    self.observers
                        .post_exec_child_all(state, input, &ExitKind::Ok)
                        .expect("Failed to run post_exec on observers");

                    libc::_exit(0);

                    Ok(ExitKind::Ok)
                }
                Ok(ForkResult::Parent { child }) => {
                    // Parent
                    // log::trace!("from parent {} child is {}", std::process::id(), child);
                    self.shmem_provider.post_fork(false)?;

                    let res = waitpid(child, None)?;
                    log::trace!("{res:#?}");
                    match res {
                        WaitStatus::Signaled(_, signal, _) => match signal {
                            nix::sys::signal::Signal::SIGALRM
                            | nix::sys::signal::Signal::SIGUSR2 => Ok(ExitKind::Timeout),
                            _ => Ok(ExitKind::Crash),
                        },
                        WaitStatus::Exited(_, code) => {
                            if code > 128 && code < 160 {
                                // Signal exit codes
                                let signal = code - 128;
                                if signal == Signal::SigAlarm as libc::c_int
                                    || signal == Signal::SigUser2 as libc::c_int
                                {
                                    Ok(ExitKind::Timeout)
                                } else {
                                    Ok(ExitKind::Crash)
                                }
                            } else {
                                Ok(ExitKind::Ok)
                            }
                        }
                        _ => Ok(ExitKind::Ok),
                    }
                }
                Err(e) => Err(Error::from(e)),
            }
        }
    }
}

#[cfg(all(feature = "std", unix))]
impl<'a, H, OT, S, SP> InProcessForkExecutor<'a, H, OT, S, SP>
where
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    OT: ObserversTuple<S>,
    S: UsesInput,
    SP: ShMemProvider,
{
    /// Creates a new [`InProcessForkExecutor`]
    pub fn new<EM, OF, Z>(
        harness_fn: &'a mut H,
        observers: OT,
        _fuzzer: &mut Z,
        _state: &mut S,
        _event_mgr: &mut EM,
        shmem_provider: SP,
    ) -> Result<Self, Error>
    where
        EM: EventFirer<State = S> + EventRestarter,
        OF: Feedback<S>,
        S: HasSolutions + HasClientPerfMonitor,
        Z: HasObjective<Objective = OF, State = S>,
    {
        let handlers = InChildProcessHandlers::new::<Self>()?;
        Ok(Self {
            harness_fn,
            shmem_provider,
            observers,
            handlers,
            phantom: PhantomData,
        })
    }

    /// Retrieve the harness function.
    #[inline]
    pub fn harness(&self) -> &H {
        self.harness_fn
    }

    /// Retrieve the harness function for a mutable reference.
    #[inline]
    pub fn harness_mut(&mut self) -> &mut H {
        self.harness_fn
    }
}

#[cfg(all(feature = "std", unix))]
impl<'a, H, OT, S, SP> TimeoutInProcessForkExecutor<'a, H, OT, S, SP>
where
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    S: UsesInput,
    OT: ObserversTuple<S>,
    SP: ShMemProvider,
{
    /// Creates a new [`TimeoutInProcessForkExecutor`]
    #[cfg(target_os = "linux")]
    pub fn new<EM, OF, Z>(
        harness_fn: &'a mut H,
        observers: OT,
        _fuzzer: &mut Z,
        _state: &mut S,
        _event_mgr: &mut EM,
        timeout: Duration,
        shmem_provider: SP,
    ) -> Result<Self, Error>
    where
        EM: EventFirer<State = S> + EventRestarter<State = S>,
        OF: Feedback<S>,
        S: HasSolutions + HasClientPerfMonitor,
        Z: HasObjective<Objective = OF, State = S>,
    {
        let handlers = InChildProcessHandlers::with_timeout::<Self>()?;
        let milli_sec = timeout.as_millis();
        let it_value = libc::timespec {
            tv_sec: (milli_sec / 1000) as _,
            tv_nsec: ((milli_sec % 1000) * 1000 * 1000) as _,
        };
        let it_interval = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let itimerspec = libc::itimerspec {
            it_interval,
            it_value,
        };

        Ok(Self {
            harness_fn,
            shmem_provider,
            observers,
            handlers,
            itimerspec,
            phantom: PhantomData,
        })
    }

    /// Creates a new [`TimeoutInProcessForkExecutor`], non linux
    #[cfg(not(target_os = "linux"))]
    pub fn new<EM, OF, Z>(
        harness_fn: &'a mut H,
        observers: OT,
        _fuzzer: &mut Z,
        _state: &mut S,
        _event_mgr: &mut EM,
        timeout: Duration,
        shmem_provider: SP,
    ) -> Result<Self, Error>
    where
        EM: EventFirer<State = S> + EventRestarter<State = S>,
        OF: Feedback<S>,
        S: HasSolutions + HasClientPerfMonitor,
        Z: HasObjective<Objective = OF, State = S>,
    {
        let handlers = InChildProcessHandlers::with_timeout::<Self>()?;
        let milli_sec = timeout.as_millis();
        let it_value = Timeval {
            tv_sec: (milli_sec / 1000) as i64,
            tv_usec: (milli_sec % 1000) as i64,
        };
        let it_interval = Timeval {
            tv_sec: 0,
            tv_usec: 0,
        };
        let itimerval = Itimerval {
            it_interval,
            it_value,
        };

        Ok(Self {
            harness_fn,
            shmem_provider,
            observers,
            handlers,
            itimerval,
            phantom: PhantomData,
        })
    }

    /// Retrieve the harness function.
    #[inline]
    pub fn harness(&self) -> &H {
        self.harness_fn
    }

    /// Retrieve the harness function for a mutable reference.
    #[inline]
    pub fn harness_mut(&mut self) -> &mut H {
        self.harness_fn
    }
}

#[cfg(all(feature = "std", unix))]
impl<'a, H, OT, S, SP> UsesObservers for InProcessForkExecutor<'a, H, OT, S, SP>
where
    H: ?Sized + FnMut(&S::Input) -> ExitKind,
    OT: ObserversTuple<S>,
    S: UsesInput,
    SP: ShMemProvider,
{
    type Observers = OT;
}

#[cfg(all(feature = "std", unix))]
impl<'a, H, OT, S, SP> UsesObservers for TimeoutInProcessForkExecutor<'a, H, OT, S, SP>
where
    H: ?Sized + FnMut(&S::Input) -> ExitKind,
    OT: ObserversTuple<S>,
    S: UsesInput,
    SP: ShMemProvider,
{
    type Observers = OT;
}

#[cfg(all(feature = "std", unix))]
impl<'a, H, OT, S, SP> HasObservers for InProcessForkExecutor<'a, H, OT, S, SP>
where
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    S: UsesInput,
    OT: ObserversTuple<S>,
    SP: ShMemProvider,
{
    #[inline]
    fn observers(&self) -> &OT {
        &self.observers
    }

    #[inline]
    fn observers_mut(&mut self) -> &mut OT {
        &mut self.observers
    }
}

#[cfg(all(feature = "std", unix))]
impl<'a, H, OT, S, SP> HasObservers for TimeoutInProcessForkExecutor<'a, H, OT, S, SP>
where
    H: FnMut(&S::Input) -> ExitKind + ?Sized,
    S: UsesInput,
    OT: ObserversTuple<S>,
    SP: ShMemProvider,
{
    #[inline]
    fn observers(&self) -> &OT {
        &self.observers
    }

    #[inline]
    fn observers_mut(&mut self) -> &mut OT {
        &mut self.observers
    }
}

/// signal handlers and `panic_hooks` for the child process
#[cfg(all(feature = "std", unix))]
pub mod child_signal_handlers {
    use alloc::boxed::Box;
    use std::panic;

    use libafl_bolts::os::unix_signals::{ucontext_t, Signal};
    use libc::siginfo_t;

    use super::{InProcessForkExecutorGlobalData, FORK_EXECUTOR_GLOBAL_DATA};
    use crate::{
        executors::{ExitKind, HasObservers},
        inputs::UsesInput,
        observers::ObserversTuple,
    };

    /// invokes the `post_exec_child` hook on all observer in case the child process panics
    pub fn setup_child_panic_hook<E>()
    where
        E: HasObservers,
    {
        let old_hook = panic::take_hook();
        panic::set_hook(Box::new(move |panic_info| {
            old_hook(panic_info);
            let data = unsafe { &mut FORK_EXECUTOR_GLOBAL_DATA };
            if data.is_valid() {
                let executor = data.executor_mut::<E>();
                let observers = executor.observers_mut();
                let state = data.state_mut::<E::State>();
                // Invalidate data to not execute again the observer hooks in the crash handler
                let input = data.take_current_input::<<E::State as UsesInput>::Input>();
                observers
                    .post_exec_child_all(state, input, &ExitKind::Crash)
                    .expect("Failed to run post_exec on observers");

                // std::process::abort();
                unsafe { libc::_exit(128 + 6) }; // ABORT exit code
            }
        }));
    }

    /// invokes the `post_exec` hook on all observer in case the child process crashes
    ///
    /// # Safety
    /// The function should only be called from a child crash handler.
    /// It will dereference the `data` pointer and assume it's valid.
    #[cfg(unix)]
    pub(crate) unsafe fn child_crash_handler<E>(
        _signal: Signal,
        _info: &mut siginfo_t,
        _context: Option<&mut ucontext_t>,
        data: &mut InProcessForkExecutorGlobalData,
    ) where
        E: HasObservers,
    {
        if data.is_valid() {
            let executor = data.executor_mut::<E>();
            let observers = executor.observers_mut();
            let state = data.state_mut::<E::State>();
            let input = data.take_current_input::<<E::State as UsesInput>::Input>();
            observers
                .post_exec_child_all(state, input, &ExitKind::Crash)
                .expect("Failed to run post_exec on observers");
        }

        libc::_exit(128 + (_signal as i32));
    }

    #[cfg(unix)]
    pub(crate) unsafe fn child_timeout_handler<E>(
        _signal: Signal,
        _info: &mut siginfo_t,
        _context: Option<&mut ucontext_t>,
        data: &mut InProcessForkExecutorGlobalData,
    ) where
        E: HasObservers,
    {
        if data.is_valid() {
            let executor = data.executor_mut::<E>();
            let observers = executor.observers_mut();
            let state = data.state_mut::<E::State>();
            let input = data.take_current_input::<<E::State as UsesInput>::Input>();
            observers
                .post_exec_child_all(state, input, &ExitKind::Timeout)
                .expect("Failed to run post_exec on observers");
        }
        libc::_exit(128 + (_signal as i32));
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;

    use libafl_bolts::tuples::tuple_list;
    #[cfg(all(feature = "std", feature = "fork", unix))]
    use serial_test::serial;

    use crate::{
        events::NopEventManager,
        executors::{inprocess::InProcessHandlers, Executor, ExitKind, InProcessExecutor},
        inputs::{NopInput, UsesInput},
        state::NopState,
        NopFuzzer,
    };

    impl UsesInput for () {
        type Input = NopInput;
    }

    #[test]
    fn test_inmem_exec() {
        let mut harness = |_buf: &NopInput| ExitKind::Ok;

        let mut in_process_executor = InProcessExecutor::<_, _, _> {
            harness_fn: &mut harness,
            observers: tuple_list!(),
            handlers: InProcessHandlers::nop(),
            phantom: PhantomData,
        };
        let input = NopInput {};
        in_process_executor
            .run_target(
                &mut NopFuzzer::new(),
                &mut NopState::new(),
                &mut NopEventManager::new(),
                &input,
            )
            .unwrap();
    }

    #[test]
    #[serial]
    #[cfg_attr(miri, ignore)]
    #[cfg(all(feature = "std", feature = "fork", unix))]
    fn test_inprocessfork_exec() {
        use libafl_bolts::shmem::{ShMemProvider, StdShMemProvider};

        use crate::{
            events::SimpleEventManager,
            executors::{inprocess::InChildProcessHandlers, InProcessForkExecutor},
            state::NopState,
            NopFuzzer,
        };

        let provider = StdShMemProvider::new().unwrap();

        let mut harness = |_buf: &NopInput| ExitKind::Ok;
        let mut in_process_fork_executor = InProcessForkExecutor::<_, (), _, _> {
            harness_fn: &mut harness,
            shmem_provider: provider,
            observers: tuple_list!(),
            handlers: InChildProcessHandlers::nop(),
            phantom: PhantomData,
        };
        let input = NopInput {};
        let mut fuzzer = NopFuzzer::new();
        let mut state = NopState::new();
        let mut mgr = SimpleEventManager::printing();
        in_process_fork_executor
            .run_target(&mut fuzzer, &mut state, &mut mgr, &input)
            .unwrap();
    }
}

#[cfg(feature = "python")]
#[allow(missing_docs)]
/// `InProcess` Python bindings
pub mod pybind {
    use alloc::boxed::Box;

    use pyo3::{prelude::*, types::PyBytes};

    use crate::{
        events::pybind::PythonEventManager,
        executors::{inprocess::OwnedInProcessExecutor, pybind::PythonExecutor, ExitKind},
        fuzzer::pybind::PythonStdFuzzerWrapper,
        inputs::{BytesInput, HasBytesVec},
        observers::pybind::PythonObserversTuple,
        state::pybind::{PythonStdState, PythonStdStateWrapper},
    };

    #[pyclass(unsendable, name = "InProcessExecutor")]
    #[derive(Debug)]
    /// Python class for OwnedInProcessExecutor (i.e. InProcessExecutor with owned harness)
    pub struct PythonOwnedInProcessExecutor {
        /// Rust wrapped OwnedInProcessExecutor object
        pub inner: OwnedInProcessExecutor<PythonObserversTuple, PythonStdState>,
    }

    #[pymethods]
    impl PythonOwnedInProcessExecutor {
        #[new]
        fn new(
            harness: PyObject,
            py_observers: PythonObserversTuple,
            py_fuzzer: &mut PythonStdFuzzerWrapper,
            py_state: &mut PythonStdStateWrapper,
            py_event_manager: &mut PythonEventManager,
        ) -> Self {
            Self {
                inner: OwnedInProcessExecutor::new(
                    Box::new(move |input: &BytesInput| {
                        Python::with_gil(|py| -> PyResult<()> {
                            let args = (PyBytes::new(py, input.bytes()),);
                            harness.call1(py, args)?;
                            Ok(())
                        })
                        .unwrap();
                        ExitKind::Ok
                    }),
                    py_observers,
                    py_fuzzer.unwrap_mut(),
                    py_state.unwrap_mut(),
                    py_event_manager,
                )
                .expect("Failed to create the Executor"),
            }
        }

        #[must_use]
        pub fn as_executor(slf: Py<Self>) -> PythonExecutor {
            PythonExecutor::new_inprocess(slf)
        }
    }

    /// Register the classes to the python module
    pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PythonOwnedInProcessExecutor>()?;
        Ok(())
    }
}
