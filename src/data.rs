#[derive(Debug)]
pub struct Data {
    pub data: Vec<Vec<f64>>,
    pub eval_data: Vec<Vec<f64>>,
    pub inputsize: i32,
    pub trainingsize: i32,
    pub evaluationsize: i32
}


impl Data {
    
    pub fn training_data(&self, _index: usize) -> Vec<f64>{
        vec![]
    }
    
    pub fn eval_data(&self, _index: usize) -> Vec<f64>{
        vec![]
    }
} 