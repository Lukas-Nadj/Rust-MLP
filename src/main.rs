#[allow(unused_parens)]
mod network;
use network::MultilayerPerceptron;


fn main(){
    let mut model = MultilayerPerceptron::init( vec![10, 16, 16, 10]  );
    
    //should print the same to the network
    model.feed();
    println!("________________________________________________________________");
    model.feed();
    
    //println!("model: {:?}", model.activations[3]);
}

