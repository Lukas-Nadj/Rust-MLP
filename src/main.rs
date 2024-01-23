#[allow(unused_parens)]
mod network;


use network::MultilayerPerceptron; // specifies the network structure and method.
use network::Data; //specifies the dimensions of the training data.
use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;

fn main(){
    println!("Initializing");
    let data:Data = Data::init_csv(",", "src/mnist_train.csv", "src/mnist_test.csv", 784); //this needs to be a serde buffered reader, or deprecated entirely.

    //initalize new model.
    let mut model = MultilayerPerceptron::init( vec![784, 10, 10, 10, 10], data);

    //load model from file.
    //let mut model = MultilayerPerceptron::load_model("./model", vec![784, 10, 10, 10, 10], data);

    //save model to disk.
    save_model(&model, "./model");

    //produce model-output. outputs shouldn't change between these two.
    model.feed(1);
    model.feed(1);
}

fn save_model(model: &MultilayerPerceptron, path: &str) {
    let serialized = serde_json::to_string(&model.layers).unwrap();
    fs::write(path, serialized).expect("Unable to write file");
}

fn load_model(model: &mut MultilayerPerceptron) {
    let data = fs::read_to_string("./model").expect("Unable to read file");
    model.layers = serde_json::from_str(&data).unwrap();
}

