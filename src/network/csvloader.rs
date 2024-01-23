use std::path::Path;
use std::fs::*;
use std::io::prelude::*;
#[derive(Debug)]
pub struct CSVLoader {
    pub Data: Vec<Vec<u8>>,
}


pub fn read(path: &str) -> String{

    let mut file = File::open(path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    contents
}