use std::fs::read_dir;
use std::net::IpAddr;
use std::path::PathBuf;
#[cfg(windows)]
use std::ptr::write_volatile;

use clap::Parser;
use libafl::{
    corpus::{InMemoryOnDiskCorpus, OnDiskCorpus},
    events::SimpleEventManager,
    executors::{HmmExecutor, InterleavedExecutor, TcpExecutor, WithObservers},
    feedbacks::{MaxMapFeedback,NotFeedback,CrashFeedback},
    fuzzer::{Fuzzer, StdFuzzer},
    generators::SeedFileGenerator,
    mutators::scheduled::{havoc_mutations, StdScheduledMutator},
    observers::ConstMapObserver,
    schedulers::QueueScheduler,
    stages::mutational::StdMutationalStage,
    state::StdState,
};
#[cfg(not(feature = "tui"))]
use libafl::monitors::SimpleMonitor;
#[cfg(feature = "tui")]
use libafl::monitors::tui::{TuiMonitor, ui::TuiUI};
use libafl::feedbacks::NeverInterestingFeedback;
use libafl_bolts::{current_nanos, rands::StdRand, tuples::tuple_list};

/// Coverage map with explicit assignments due to the lack of instrumentation
const NUMBER_OF_NODES: usize = 51;
const SIGNALS_SIZE: usize = NUMBER_OF_NODES * NUMBER_OF_NODES;
static mut SIGNALS: [u8; SIGNALS_SIZE] = [0; SIGNALS_SIZE];
static mut SIGNALS_PTR: *mut u8 = unsafe { SIGNALS.as_mut_ptr() };


#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    cleanup_script_path: PathBuf,
    #[arg(long)]
    target_script_path: PathBuf,
    #[arg(long, allow_hyphen_values = true)]
    target_options: String,
    #[arg(long, default_value = "./crashes")]
    objective_dir: PathBuf,
    #[arg(long, default_value = "./queue")]
    corpus_dir: PathBuf,
    #[arg(long, default_value = "10000")]
    target_startup_usecs: u64,
    #[arg(long, default_value = "seeds")]
    seed_files_folder_path: PathBuf,
    #[arg(long)]
    hmm_lib_path: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    target_ip: IpAddr,
    #[arg(long, default_value = "8082")]
    target_port: u16,
}

#[allow(clippy::similar_names, clippy::manual_assert)]
pub fn main() {
    let args = Args::parse();
    let cleanup_script_path = Some(args.cleanup_script_path);
    let target_path = Some(args.target_script_path);
    let target_options = Some(args.target_options);
    let target_startup_usecs = Some(args.target_startup_usecs);
    let corpus_dir = Some(args.corpus_dir);
    let objective_dir = Some(args.objective_dir);
    let seed_files_folder = Some(args.seed_files_folder_path.as_path());

    //----------------
    // HMMs
    //----------------

    // Create an observation channel using the signals map
    let observer_hmm = unsafe { ConstMapObserver::<u8, SIGNALS_SIZE>::from_mut_ptr("signals", SIGNALS_PTR) };
    //let observer = unsafe { StdMapObserver::from_mut_ptr("signals", SIGNALS_PTR, SIGNALS.len()) };

    // Feedback to rate the interestingness of an input
    let mut feedback = MaxMapFeedback::new(&observer_hmm);

    // A feedback to choose if an input is a solution or not
    // The TCP executor will execute a ping command to check whether the target is still alive and will set the exit code accordingly. As a result, we can use the normal crash feedback to evaluate the behavior of the target.
    let mut objective = CrashFeedback::new();

    // create a State from scratch
    let mut state = StdState::new(
        // RNG
        StdRand::with_seed(current_nanos()),
        // Corpus that will be evolved, written to disk to be able to analyze it later on (e.g. using profuzz bench)
        InMemoryOnDiskCorpus::new(corpus_dir.expect("At least the default value for corpus directory should be set.")).unwrap(),
        // Corpus in which we store solutions (crashes in this example),
        // on disk so the user can get them after stopping the fuzzer
        OnDiskCorpus::new(objective_dir.expect("At least the default value for crash directory should be set.")).unwrap(),
        // States of the feedbacks.
        // The feedbacks can report the data that should persist in the State.
        &mut feedback,
        // Same for objective feedbacks
        &mut objective,
    )
        .unwrap();

    // The Monitor trait define how the fuzzer stats are displayed to the user
    #[cfg(not(feature = "tui"))]
    let mon = SimpleMonitor::new(|s| println!("{s}"));
    #[cfg(feature = "tui")]
    let ui = TuiUI::with_version(String::from("Baby Fuzzer"), String::from("0.0.1"), false);
    #[cfg(feature = "tui")]
    let mon = TuiMonitor::new(ui);

    // The event manager handle the various events generated during the fuzzing loop
    // such as the notification of the addition of a new item to the corpus
    let mut mgr = SimpleEventManager::new(mon);

    // A queue policy to get testcases from the corpus
    let scheduler = QueueScheduler::new();

    // A fuzzer with feedbacks and a corpus scheduler
    let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);

    // Create the executor for a tcp target
    let tcp_executor = TcpExecutor::new(
        args.target_ip,
        args.target_port,
        cleanup_script_path,
        target_path,
        target_options,
        target_startup_usecs,
    );

    let tcp_executor_with_observers = WithObservers::new(
        tcp_executor,
        tuple_list!(),
    );

    let hmm_executor = unsafe {
        HmmExecutor::new(
            SIGNALS_PTR,
            args.target_port,
            args.hmm_lib_path,
            None,
            None,
        )
    };

    let hmm_executor_with_observers = WithObservers::new(
        hmm_executor,
        tuple_list!(observer_hmm),
    );

    // hmm executor needs to be first to be able to start the network traffic recording before the actual tcp testcase is sent
    let mut executor = InterleavedExecutor::new(hmm_executor_with_observers, tcp_executor_with_observers, &mgr, &fuzzer);

    // Generator using the seeds from the seed file
    let mut generator = SeedFileGenerator::new(seed_files_folder.unwrap());

    // check how many seeds we have and create the initial inputs from that
    let number_of_seeds = read_dir(seed_files_folder.unwrap())
        .expect(&format!("There should be at least one seed file."))
        .into_iter()
        .map(|x| x.expect(""))
        .filter(|x| x.path().to_str().expect("Should be able to create a string from the path.").ends_with("raw"))
        .count();
    state
        .generate_initial_inputs(&mut fuzzer, &mut executor, &mut generator, &mut mgr, number_of_seeds)
        .expect("Failed to generate the initial corpus");

    // Setup a mutational stage with a basic bytes mutator
    let mutator = StdScheduledMutator::new(havoc_mutations());
    let mut stages = tuple_list!(StdMutationalStage::new(mutator));

    fuzzer
        // .fuzz_loop_for(&mut stages, &mut executor, &mut state, &mut mgr, 10) // fuzz only a fixed number of rounds
        .fuzz_loop(&mut stages, &mut executor, &mut state, &mut mgr) // infinitely
        .expect("Error in the fuzzing loop");
}