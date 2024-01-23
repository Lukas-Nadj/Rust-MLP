use serde::{Deserialize, Serialize};
use crate::network::csvloader;
use super::csvloader::*;

#[derive(Serialize, Deserialize, Debug)]
pub struct Input{
    pub label: f64,
    pub data: Vec<f64>,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct Data {
    pub data: Vec<Input>,
    pub eval_data: Vec<Input>,
    pub inputsize: i32,
    pub trainingsize: i32,
    pub evaluationsize: i32
}


impl Data {

    pub fn init_csv(delimiter: &str, data_path: &str, eval_path: &str, inputsize:i32) -> Self{
        //training_data
        let binding = csvloader::read(data_path);
        let lines = binding.split("\n");
        let training_size = lines.clone().count()  as i32;
        let mut data:Vec<Input> = Vec::new();
        for line in lines.skip(1) {
            let mut inputs:Vec<f64> = Vec::new();
            let row = line.split(delimiter).collect::<Vec<&str>>();
            for i in 1..row.len() {
                    inputs.push(row[i].trim().parse::<f64>().unwrap_or(0.0f64));
            }
            data.push(Input{label: row[0].parse::<f64>().unwrap_or(0.0f64), data:inputs})
        }

        //eval_data
        let binding =  csvloader::read(eval_path);
        let eval_lines = binding.split("\n");
        let evaluation_size = eval_lines.clone().count() as i32;
        let mut eval_data:Vec<Input> = Vec::new();
        for eline in eval_lines.skip(1) {
            let mut inputs:Vec<f64> = Vec::new();
            let row = eline.split(delimiter).collect::<Vec<&str>>();
            for i in 1..row.len() {
                inputs.push(row[i].trim().parse::<f64>().unwrap_or(0.0f64));
            }
            eval_data.push(Input{label: row[0].parse::<f64>().unwrap_or(0.0f64), data:inputs})
        }

        Data {data, eval_data, inputsize, trainingsize: training_size, evaluationsize: evaluation_size}
    }

    pub fn training_data(&self, _index: usize) -> Vec<f64>{
        vec![]
    }
    
    pub fn eval_data(&self, _index: usize) -> Vec<f64>{
        vec![]
    }
} 