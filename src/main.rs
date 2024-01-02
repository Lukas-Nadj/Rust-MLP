#[allow(unused_parens)]
mod network;
mod data;
use network::MultilayerPerceptron; // specifies the network structure and method.
use data::Data; //specifies the dimensions of the traning data.
//use CSVloader //specifies the loading of data into memory.
//use network::modelfs; // Methods for saving and loading the model.


fn main(){
    
    let data:Data = Data{data: vec![vec![0.0f64]], eval_data: vec![vec![0.0f64]], inputsize: 10, trainingsize: 2, evaluationsize: 1};
    let mut model = MultilayerPerceptron::init( vec![784, 16, 16, 16, 16, 10], &data);
    
    //should print the same to the network
    model.feed();
    model.feed();

    //println!("model: {:?}", model.activations[3]);
}

