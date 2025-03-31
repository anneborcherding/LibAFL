//! An `InterleavedExecutor` interweaves a primary executor and a secondary one.
//! In comparison to the [`crate::executors::Combined`] it runs both executors in `run_target`, in an interweaved fashion.
//! It will run the executors in the following order: (I) `primary.run_target`, (II) `secondary.run_target`, (III) `primary.post_run_reset`, (IV) `secondary.post_run_reset`.

use core::fmt::Debug;

use crate::{
    executors::{Executor, ExitKind, HasObservers},
    observers::UsesObservers,
    state::{HasExecutions, UsesState},
    Error,
};

/// An [`InterleavedExecutor`] interweaves a primary executor with a secondary one
#[derive(Debug)]
pub struct InterleavedExecutor<A, B> {
    primary: A,
    secondary: B,
}

impl<A, B> InterleavedExecutor<A, B> {
    /// Create a new `InterleavedExecutor`, interweaving the given `executor`s.
    pub fn new<EM, Z>(primary: A, secondary: B, _mgr: &EM, _fuzzer: &Z) -> Self
    where
        A: Executor<EM, Z>,
        B: Executor<EM, Z, State = A::State>,
        EM: UsesState<State = A::State>,
        Z: UsesState<State = A::State>,
    {
        Self { primary, secondary }
    }

    /// Retrieve the primary `Executor` that is wrapped by this `InterleavedExecutor`.
    pub fn primary(&mut self) -> &mut A {
        &mut self.primary
    }

    /// Retrieve the secondary `Executor` that is wrapped by this `InterleavedExecutor`.
    pub fn secondary(&mut self) -> &mut B {
        &mut self.secondary
    }
}

impl<A, B, EM, Z> Executor<EM, Z> for InterleavedExecutor<A, B>
where
    A: Executor<EM, Z>,
    B: Executor<EM, Z, State = A::State>,
    EM: UsesState<State = A::State>,
    EM::State: HasExecutions,
    Z: UsesState<State = A::State>,
{
    fn run_target(
        &mut self,
        fuzzer: &mut Z,
        state: &mut Self::State,
        mgr: &mut EM,
        input: &Self::Input,
    ) -> Result<ExitKind, Error> {
        *state.executions_mut() += 1;

        let ret1 = self.primary.run_target(fuzzer, state, mgr, input)?;
        let ret2 = self.secondary.run_target(fuzzer, state, mgr, input)?;
        self.primary.post_run_reset();
        self.secondary.post_run_reset();

        match (ret1, ret2) {
            (ExitKind::Ok, ExitKind::Ok) => Ok(ExitKind::Ok),
            (ExitKind::Crash, _) => Ok(ExitKind::Crash),
            (_, ExitKind::Crash) => Ok(ExitKind::Crash),
            (ExitKind::Timeout, _) => Ok(ExitKind::Timeout),
            (_, ExitKind::Timeout) => Ok(ExitKind::Timeout),
            _ => Ok(ExitKind::Crash),
        }
    }
}

impl<A, B> UsesState for InterleavedExecutor<A, B>
where
    A: UsesState,
{
    type State = A::State;
}

impl<A, B> UsesObservers for InterleavedExecutor<A, B>
where
    A: UsesObservers,
{
    type Observers = A::Observers;
}

impl<A, B> HasObservers for InterleavedExecutor<A, B>
where
    A: HasObservers,
{
    #[inline]
    fn observers(&self) -> &Self::Observers {
        self.primary.observers()
    }

    #[inline]
    fn observers_mut(&mut self) -> &mut Self::Observers {
        self.primary.observers_mut()
    }
}
