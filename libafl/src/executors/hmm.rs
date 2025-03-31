///! The [`HmmExecutor`] is an executor that will trigger an HMM to approximate the behavior of the target with respect to the given input.

use crate::executors::{Executor, ExitKind};
use crate::inputs::BytesInput;
use crate::prelude::UsesInput;
use crate::state::UsesState;
use core::fmt::{Debug, Formatter};
use libafl_bolts::Error;
use pyo3::prelude::PyModule;
use pyo3::types::PyList;
use pyo3::{PyAny, Python};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::ptr::write;
use std::{fs, thread};
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::thread::JoinHandle;
use std::time::SystemTime;
use std::vec::Vec;
use pcap::{Capture, Device};

const DELETE_FILES_IMMEDIATELY: bool = true;


/// Executor that triggers a Hidden Markov Model (HMM) to approximate the behavior of the target with respect to the given input.
pub struct HmmExecutor<S>
    where
        S: UsesInput,
{
    observation_map: *mut u8,
    target_port: u16,
    thread_signal_sender: Option<Sender<()>>,
    thread_signal_receiver: Option<Receiver<()>>,
    thread_handle: Option<JoinHandle<()>>,
    hmm_lib_path: PathBuf,
    config_path: PathBuf,
    pcap_dump_path: PathBuf,
    curr_pcap_path: PathBuf,
    phantom: PhantomData<S>,
}

impl<S> HmmExecutor<S>
    where
        S: UsesInput<Input=BytesInput>,
{
    /// Creates a new HmmExecutor which will approximate the behavior of the target
    pub fn new(
        observation_map: *mut u8,
        target_port: u16,
        hmm_lib_path: PathBuf,
        config_path: Option<PathBuf>,
        pcap_dump_path: Option<PathBuf>,
    ) -> Self {
        pyo3::prepare_freethreaded_python();
        let config_path =
            config_path.unwrap_or(PathBuf::from(&*hmm_lib_path.join("config.yml")));
        let pcap_dump_path = pcap_dump_path.unwrap_or(PathBuf::from(&*hmm_lib_path.join("dump")));
        let mut hmm_executor = Self {
            observation_map,
            target_port,
            phantom: Default::default(),
            thread_signal_sender: None,
            thread_signal_receiver: None,
            thread_handle: None,
            hmm_lib_path,
            config_path,
            pcap_dump_path,
            curr_pcap_path: Default::default(),
        };
        hmm_executor.init_hmms();
        hmm_executor
    }

    fn init_hmms(&mut self) {
        Python::with_gil(|py| {
            let sys: &PyAny = py.import("sys").expect("").getattr("path").expect("");
            let sys: &PyList = pyo3::PyTryInto::try_into(sys).unwrap();
            sys.insert(0, self.hmm_lib_path.clone()).expect("");
            let path = py.import("sys").expect("").getattr("path").expect("");
            println!("--- python connection related paths ---");
            println!("sys.path: {:?}", path);
            println!("Path to HMM python folder: {:?}", self.hmm_lib_path);
            println!("Path to HMM config file: {:?}", self.config_path);
            println!("Path to intermediate PCAP file: {:?}", self.pcap_dump_path);
            println!("--- end of python connection related paths ---");
            let hmm_coverage = PyModule::import(py, "scripts.hmm_coverage").expect("Was not able to load the python hmm module. It might help to check your paths (see above).");
            let args = (self.config_path.to_str(), );
            hmm_coverage.getattr("init").expect("").call1(args).expect("It should be possible to configure the HMMs. Check your config path if you receive a FileNotFoundError.");
        });
    }
}

impl<S> Debug for HmmExecutor<S>
    where
        S: UsesInput,
{
    #[allow(unused_variables)]
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("HmmExecutor")
            .field(&self.observation_map)
            .finish()
    }
}

impl<S> UsesState for HmmExecutor<S>
    where
        S: UsesInput,
{
    type State = S;
}

impl<EM, Z, S> Executor<EM, Z> for HmmExecutor<S>
    where
        EM: UsesState<State=S>,
        Z: UsesState<State=S>,
        S: UsesInput<Input=BytesInput>,
{
    #[allow(unused_variables)]
    fn run_target(
        &mut self,
        fuzzer: &mut Z,
        state: &mut Self::State,
        mgr: &mut EM,
        input: &Self::Input,
    ) -> Result<ExitKind, Error> {
        self.curr_pcap_path = self.pcap_dump_path.join(Path::new(&format!("{:?}.pcap", &SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).expect("").as_secs())).as_os_str());
        let (sender_thread, receiver_main) = mpsc::channel();
        let (sender_main, receiver_thread) = mpsc::channel(); // TODO would be nicer to only need one channel
        let pcap_path = self.curr_pcap_path.clone();
        let port = self.target_port;

        let handle = thread::spawn(move || {
            let device = Device::list().unwrap().iter().filter(|x| x.name=="lo").collect::<Vec<&Device>>()[0].clone();
            let mut cap = Capture::from_device(device).unwrap()
                .promisc(true)
                .snaplen(5000)
                .timeout(50)
                .open().unwrap();
            let filter = format!("tcp and port {}", port);
            cap.filter(&filter, false).expect("Should be able to set the filter and the filter should have correct syntax.");

            let device2 = Device::list().unwrap().iter().filter(|x| x.name=="lo").collect::<Vec<&Device>>()[0].clone();
            let cap2 = Capture::from_device(device2).unwrap()
                .promisc(true)
                .snaplen(5000)
                .open().unwrap();
            let mut savefile = cap.savefile(pcap_path).unwrap();

            sender_thread.send(()).expect("Should be able to send a message."); // Finished setup

            let expected_number_of_packets = 8; // TODO this is hacky, but right now I don't think there really is a better version to do this, since a non-blocking iterator for packets is not possible due to the restrictions of libpcap. With this, the stop message from the main thread is technically not necessary.
            let mut count = 0;
            loop {
                match cap.next_packet() {
                    Err(_) => {}
                    Ok(pkt) => {
                        savefile.write(&pkt);
                        count = count + 1;
                    }
                };
                if count >= expected_number_of_packets {
                    match receiver_thread.try_recv() {
                        Ok(_) | Err(TryRecvError::Disconnected) => {
                            savefile.flush().expect("Should be able to flush the savefile."); // making sure that all packets are actually written to the file
                            break;
                        },
                        Err(TryRecvError::Empty) => {},
                    }
                }
            }

        });
        self.thread_handle = Option::from(handle);
        self.thread_signal_sender = Option::from(sender_main);

        // wait for the thread to setup everything
        loop {
            match receiver_main.try_recv() {
                Ok(_) | Err(TryRecvError::Disconnected) => {
                    break;
                },
                Err(TryRecvError::Empty) => {},
            }
        }
        self.thread_signal_receiver = Option::from(receiver_main);
        Ok(ExitKind::Ok)
    }

    /// Stop the traffic recording
    fn post_run_reset(&mut self) {
        match &self.thread_signal_sender {
            Some(sender) => {
                sender.send(()).expect("Should be able to send the message on the thread channel.");
            }
            None => {}
        }
        if let Some(handle) = self.thread_handle.take() {
            handle.join().expect("failed to join thread");
        }

        Python::with_gil(|py| {
            let sys: &PyAny = py.import("sys").expect("").getattr("path").expect("");
            let sys: &PyList = pyo3::PyTryInto::try_into(sys).unwrap();
            sys.insert(0, self.hmm_lib_path.to_str()).expect("");
            let _path = py.import("sys").expect("").getattr("path").expect("");
            // let dummy = PyModule::import(py, "scripts.test").expect("");
            // let result = dummy.getattr("test").expect("").call0().expect("");
            let hmm_cov = PyModule::import(py, "scripts.hmm_coverage").expect("");

            // TODO would be great to suppress the reading from file info, did look into options to suppress this in python but all the things I tried did not work (context manager; maybe because the output comes from tcpdump which is spawned in a subprocess?)
            let args = (self.curr_pcap_path.to_str(), );
            let result = hmm_cov
                .getattr("calculate_coverage")
                .expect("Was not able to load the hmm coverage function. It might help to check your paths (see above), or to activate the conda environment for python.")
                .call1(args)
                .expect("Got an error while calling the hmm coverage function. It might help to check your paths (see above), or to activate the conda environment for python.");
            let res: Vec<i64> = result.extract().unwrap();
            // println!("{:?}", res);
            for (i, j) in res.iter().enumerate() {
                if *j == 1 {
                    unsafe { write(self.observation_map.add(i), 1) }
                }
            }
        });

        if DELETE_FILES_IMMEDIATELY {
            let _ = fs::remove_file(self.curr_pcap_path.clone());
        }
    }
}
