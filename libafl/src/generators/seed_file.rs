use std::{fs::File, io::prelude::*, marker::PhantomData, path::Path, vec::Vec};
use std::fs::read_dir;

use crate::{Error, generators::Generator, prelude::BytesInput};

/// Generator that takes the seed inputs from a seed file.
#[derive(Clone, Debug)]
pub struct SeedFileGenerator<S> {
    index: usize,
    phantom: PhantomData<S>,
    content: Vec<BytesInput>,
}

impl<S> SeedFileGenerator<S> {
    /// new generator that takes the seed inputs from the files at the given path.
    pub fn new(folder: &Path) -> Self {
        let paths = read_dir(folder).expect(&format!("There should be at least one seed file at {:?}", folder));
        let mut content: Vec<Vec<u8>> = vec!();
        for path_res in paths {
            let path = path_res.unwrap().path();
            if path.to_str().expect("Should be able to convert this path to a string.").ends_with(".raw")
            {
                let mut file = File::open(path.clone()).expect(&format!("Was not able to read the file which is supposed to hold a seed in {:?}", path));
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer).expect("");
                content.push(buffer);
            }
        }
        let result = content.into_iter().map(|x| x.into()).collect();
        Self { index: 0, phantom: Default::default(), content: result }
    }
}

impl<S> Generator<BytesInput, S> for SeedFileGenerator<S> {
    fn generate(&mut self, _state: &mut S) -> Result<BytesInput, Error> {
        if self.index < self.content.len() {
            let val = self.content[self.index].clone();
            self.index = self.index + 1;
            return Ok(val);
        }
        Err(Error::empty(
            "No more items in iterator when generating inputs",
        ))
    }
}
