///! The [`TcpExecutor`] is an executor that will send the input via TCP to the target.
use crate::executors::{Executor, ExitKind};
use crate::inputs::BytesInput;
use crate::prelude::UsesInput;
use crate::state::UsesState;
use crate::std::io::Write;
use core::fmt::{Debug, Formatter};
use libafl_bolts::Error;
use run_script;
use std::{fs, io};
use std::marker::PhantomData;
use std::net::{IpAddr, Shutdown, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use alloc::string::String;
use std::io::{BufRead, Read};
use std::prelude::v1::Vec;
use std::thread::sleep;
use std::time::Duration;

/// Executor that sends the input via TCP to the target.
pub struct TcpExecutor<S>
where
    S: UsesInput,
{
    target_ip_addr: IpAddr,
    target_port: u16,
    cleanup_script: Option<PathBuf>,
    target_path: Option<PathBuf>,
    target_process: Option<Child>,
    target_options: Option<String>,
    target_startup_usecs: Option<u64>,
    connection_retries: u8,
    phantom: PhantomData<S>
}

impl<S> TcpExecutor<S>
where
    S: UsesInput<Input = BytesInput>,
{
    /// Creates a new TcpExecutor which will send the input to the given IP address and port via tcp.
    /// If you want the executor to run a cleanup script after each run, you can specify the script's path.
    /// If you want the executor to start the target, you can specify the target's executable path
    pub fn new(target_ip_addr: IpAddr, target_port: u16, cleanup_script: Option<PathBuf>, target_path: Option<PathBuf>, target_options: Option<String>, target_startup_usecs: Option<u64>) -> Self {
        Self {
            target_ip_addr,
            target_port,
            cleanup_script,
            target_path,
            target_options,
            target_startup_usecs,
            target_process: None,
            connection_retries: 3,
            phantom: Default::default(),
        }
    }

    // TODO we should probably drop the child process somewhere, this is not handled by the Command crate (even if the child goes out of scope)
    fn start_target(&mut self) {
        if let Some(target_path) = &self.target_path {
            if let Some(target_options) = &self.target_options {
                let args: Vec<_> = target_options.split(" ").collect();
                self.target_process = Some(Command::new(target_path).args(args).spawn().expect("Target failed to start."));
            }
            else {
                self.target_process = Some(Command::new(target_path).spawn().expect("Target failed to start."));
            }
            println!("Waiting for target_startup_usecs ({} microseconds) ...", self.target_startup_usecs.unwrap_or(10000));
            sleep(Duration::from_micros(self.target_startup_usecs.unwrap_or(10000)));
        }

    }

    fn restart_target(&mut self) {
        println!("Restarting target...");
        if let Some(ref mut child) = self.target_process {
            child.kill().expect("Should be able to kill the target process successfully. Process lib handles the case of already exited processes.");
        }
        self.start_target();
    }

    fn monitor_target(&mut self) -> Result<ExitKind, Error>{
        let status = Command::new("ping")
            .arg("-c")
            .arg("1")
            .arg("-W")
            .arg("1")
            .arg(format!("{}", self.target_ip_addr))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("ping command did not start.");

        println!("Run ping. It finished with status code {:?}", status.code());

        match status.code() {
            Some(0) => Ok(ExitKind::Ok),
            Some(1) => {
                // restart target if it does not respond to ping
                println!("Restarting target since it did not respond to ping.");
                self.restart_target();
                Ok(ExitKind::Crash)
            },
            Some(2) => Err(Error::illegal_state(String::from("An error occurred while executing ping.")
            )),
            Some(code) => Err(Error::illegal_state(String::from(format!("Ping finished with unusual exit code: {}.", code))
            )),
            None => Err(Error::illegal_state(String::from("Ping command was interrupted by signal.")
            )),
        }
    }

    fn try_connect(&mut self) -> Option<TcpStream> {
        let mut stream_opt = None;
        for _ in 0..self.connection_retries {
            let result = TcpStream::connect(format!("{}:{}", self.target_ip_addr, self.target_port));

            match result {
                Ok(mut tcp_stream) => {
                    let mut read_str= String::new();
                    tcp_stream.set_read_timeout(Option::from(Duration::from_secs(5))).expect("");
                    if let Ok(_) = tcp_stream.read_to_string(&mut read_str) {
                        println!("I read a string!{}", read_str);
                    }
                    stream_opt = Some(tcp_stream);
                    break
                }
                Err(r) => {
                    println!("Received an error while trying to connect to the target: {:?}", r);
                    self.restart_target()
                }
            }
        }
        stream_opt
    }

    fn split_bytes_at_crlf(bytes: &[u8]) -> Vec<&[u8]> {
        let mut result = Vec::new();
        let mut start = 0;

        while let Some(mut pos) = bytes[start..].windows(2).position(|w| w == [13, 10]) {
            pos += start; // Adjust position to global index
            result.push(&bytes[start..pos+2]);
            start = pos + 2; // Move start to next position after CRLF
        }

        // Add the remaining bytes after the last CRLF if any
        if start < bytes.len() {
            result.push(&bytes[start..]);
        }

        result
    }

    fn write_input(&mut self, mut tcp_stream: &TcpStream, input: BytesInput) {
        for input in Self::split_bytes_at_crlf(&input.bytes) {
            // TODO: This is FTP specific
            println!("sending input {:?}", input);
            let tcp_result = tcp_stream
                .write(&*input);

            match tcp_result {
                Ok(_) => {}
                Err(error) => {println!("Got an error while trying to write on the TCP stream. {:?}", error)}
            }
            // wait for answer
            let mut reader = io::BufReader::new(&mut tcp_stream);
            let received = reader.fill_buf();
            if let Ok(received_content) = received {
                let received_len = received_content.len();
                if received_len > 0 {
                    let data = String::from_utf8(received_content.to_vec())
                        .map(|msg| println!("{}", msg))
                        .map_err(|_| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "Couldn't parse received string as utf8",
                            )
                        });
                    println!("I read a string as a response!{:?}", data.unwrap_or(()));
                    reader.consume(received_len);
                } else {
                    println!("Did not read a response ;(");
                }
            }
            else {
                println!("Received not ok. {:?}", received);
            }
        }

    }
}



impl<S> Debug for TcpExecutor<S>
where
    S: UsesInput,
{
    #[allow(unused_variables)]
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("TcpExecutor")
            .field(&self.target_ip_addr)
            .field(&self.target_port)
            .finish()
    }
}

// TODO do we want to implement our own Input for TCP based input? maybe.
impl<S> UsesState for TcpExecutor<S>
where
    S: UsesInput,
{
    type State = S;
}

impl<EM, Z, S> Executor<EM, Z> for TcpExecutor<S>
where
    EM: UsesState<State = S>,
    Z: UsesState<State = S>,
    S: UsesInput<Input = BytesInput>,
{

    #[allow(unused_variables)]
    fn run_target(
        &mut self,
        fuzzer: &mut Z,
        state: &mut Self::State,
        mgr: &mut EM,
        input: &Self::Input,
    ) -> Result<ExitKind, Error> {

        // only start target the first time
        // restarting the target after a crash is handled later on during the processing of ping results
        if let Some(ref mut child) = self.target_process {
            match child.try_wait() {
                // child exited
                Ok(Some(status)) => {
                    println!("Target process seems to have exited with status {:?}. For LightFTP, this is expected in the patched version.", status);
                    self.restart_target()
                },
                // child still running
                Ok(None) => (),
                // some error occurred, better to restart target
                Err(r) => {
                    println!("There was an error while calling try_wait on the target process: {:?}", r);
                    // self.restart_target()
                }
            }
        }
        else {
            self.start_target();
        }

        let stream_opt = self.try_connect();

        if let Some(ref tcp_stream) = stream_opt {
            self.write_input(&tcp_stream, input.clone());
        } else
        {
            return Err(Error::illegal_state(String::from(format!("Was not able to connect to target after {} retries.", self.connection_retries))))
        }

        let monitoring_result = self.monitor_target();

        if let Some(ref tcp_stream) = stream_opt {
            // need to do this after monitoring since server might shut down after this (on purpose)
            tcp_stream.shutdown(Shutdown::Both).expect("Failed to shutdown TcpStream.");
        }

        monitoring_result
    }

    /// Run the given cleanup script, similar to AFLNet
    fn post_run_reset(&mut self) {
        if let Some(path) = &self.cleanup_script {
            let script = fs::read_to_string(path.as_path())
                .expect(&*format!("Was not able to read the file which is supposed to hold the cleanup script in {:?}", path.to_str()));
            let options = run_script::ScriptOptions::new();
            let args = vec![];
            let (_code, _output, _error) = run_script::run(&script, &args, &options).unwrap();
            // TODO log the result somewhere?
        }
    }
}