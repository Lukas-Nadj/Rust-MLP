use std::fs;
use rand::prelude::*;
use super::data::Data;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Layer { //:3
    weights:  Vec<Vec<f64>>,
    bias: Vec<f64>,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct MultilayerPerceptron{
    #[serde(skip_serializing)]
    pub data: Data,
    #[serde(skip_serializing)]
	pub activations: Vec<Vec<f64>>,
    #[serde(skip_serializing)]
    pub pre_activations: Vec<Vec<f64>>,
	pub layers: Vec<Layer>, //could become a vec array of touples, or be defined inline without Layer struct. but for now that would be a headache.
    #[serde(skip_serializing)]
    pub model_struct: Vec<i32> //could simplify some loops later
}

pub enum BatchingType { 
    MiniBatch(u32),
    Batching,
    Online
}

impl<'a> MultilayerPerceptron {
    pub fn load( model_fs:i32 , data:&'a Data ) /* -> Self */{
        
    }
}

impl<'a> MultilayerPerceptron { //ni
    pub fn load_model(path: &str, model_struct: Vec<i32>, data:Data) -> Self{
        let file = fs::read_to_string(path).expect("Unable to read file");
        let layers = serde_json::from_str(&file).unwrap();

        let mut activations: Vec<Vec<f64>> = vec![vec![]; model_struct.len()]; //initialize array size to model size
        let mut pre_activations: Vec<Vec<f64>> = vec![vec![]; model_struct.len()];
        let mut layers: Vec<Layer> = layers;

        //initialize activations array
        for (i, size) in model_struct.iter().enumerate() {
            activations[i].resize(*size as usize, 0.0f64);
        }

        Self {data, activations, pre_activations, layers, model_struct}
    }

	pub fn init( model_struct: Vec<i32>, data:Data ) -> Self{	 //TODO: assert data.data has the correct dimensions. with input layer.
        let mut activations: Vec<Vec<f64>> = vec![vec![]; model_struct.len()]; //initialize array size to model size
        let mut pre_activations: Vec<Vec<f64>> = vec![vec![]; model_struct.len()];
        let mut layers: Vec<Layer> = (0..model_struct.len()).map(|_| Layer { 
            weights: vec![vec![]],
            bias: vec![],
        }).collect();
        
        //initialize activations array
        let mut rng = rand::thread_rng();
        for (i, size) in model_struct.iter().enumerate() { 
            activations[i].resize(*size as usize, 0.0f64); 
        }
        
        //initialize weights and biases with random floats.
        for (l, size) in model_struct.iter().enumerate() { 
            if l>0 {layers[l].weights.resize(*size as usize, Vec::new())};
            if l>0 {layers[l].bias.resize(*size as usize, 0.0001f64)};
            pre_activations[l].resize(activations[l].len(), 0.0f64); //weighted_sum + bias; probably a dumb way to initialize it but whatever.
            for i in 0..*size as usize {
                if(l>0){
                    //i fucking love closures and resize_with !!!
                    layers[l].weights[i].resize_with(activations[l-1].len(), || rng.gen::<f64>()-0.5f64);
                }
            }
        }
        //return a new model struct that takes ownership of these variables. //nope, we should just use the Self struct instead. now we can reinitialize a model without creating a new struct? really not significant for an initialization..
        Self {data, activations, pre_activations, layers, model_struct}
    }


    pub fn cost(&self, activations: &Vec<f64>, target: &Vec<f64>) -> f64{  
        //not done, and not useful. //how bout NOW benson
        let mut sum = 0.0f64;
        let mut cost: Vec<f64> = vec![0.0; target.len()];
        
        for (index, _err) in activations.iter().enumerate() {
            cost[index] = (activations[index] - target[index]).powi(2);
            sum += (activations[index] - target[index]).powi(2);
        }
        
        sum
    }

    pub fn dot(left: &Vec<Vec<f64>>, right: &Vec<f64>) -> Vec<f64> { //so far mostly for weights[][] dot activations[], not generalized.
        //new vec<f64> for the result of the operation
        let mut output:Vec<f64> = Vec::new();
        output.resize(left.len(), 0.0f64);
        
        //get array len's to a temp variable so we don't need to retrieve it every iteration of the subsequent loop.
        let len = left.len();
        let len_inner = left[0].len();
        
        for row in 0..len {
            let mut sum = 0.0f64;
            
            for i in 0..len_inner {
                //Z += W*A[l], weighted sum for each neuron/row
                sum += left[row][i]*right[i];
            }
            
            output[row] = sum;
        }
        return output; //output would have been fine but i find this so much more readable from java ;-; might change later, doesn't really matter (i think?)
    } 
    
    pub fn add(left: &Vec<f64>, right: &Vec<f64>) -> Vec<f64> {
        //adding arrays 1 to 1 requires the same dimensions and length.
        assert_eq!(left.len(), right.len());
        assert_ne!(left.len(),  0, "Fuck");
        
        //init ouput array.
        let mut output:Vec<f64> = Vec::new();
        output.resize(left.len(), 0.0f64);
        
        if right.len() > 0 { //dunno why i wrote this, if it's dumb the compiler will remove it lol.
        for i in 0..right.len() {
            output[i] = left[i] + right[i];
        }
        }
        
        output
    }

    pub fn subtract(left: &Vec<f64>, right: &Vec<f64>) -> Vec<f64> {
        //adding arrays 1 to 1 requires the same dimensions and length.
        assert_eq!(left.len(), right.len());
        assert_ne!(left.len(),  0, "Fuck");

        //init ouput array.
        let mut output:Vec<f64> = Vec::new();
        output.resize(left.len(), 0.0f64);

        if right.len() > 0 { //dunno why i wrote this, if it's dumb the compiler will remove it lol.
            for i in 0..right.len() {
                output[i] = left[i] - right[i];
            }
        }

        output
    }
    
    pub fn sigmoid( z: &Vec<f64>) -> Vec<f64> { //applies activation function to an entire layer.
        //init
        let mut output:Vec<f64> = Vec::new();
        output.resize(z.len(), 0.0f64);
        
        //Sigmoid activation
        for i in 0..z.len() {
            output[i] = 1.0f64 / (1.0f64 + (-z[i]).exp()); //I forgot the '-' in front of z[i] imagine the rage of trying to train the model if only the derivative of the sigmoid function was correct 😬
        }
        
        output
    }

    /*pub fn dsigmoid(self, z: &Vec<f64>) -> Vec<f64> {
        let output = Vec::with_capacity(z.len());
        let sig = self.sigmoid(z);
        for l in 0..z.len() {
            output[l] = sig[l]*(1-sig[l]);
        }
        output
    }*/
    
    pub fn elementwise(&self, left: &Vec<f64>, right: &Vec<f64>, offset: f64) -> Vec<f64> {
        //init
        let mut output: Vec<f64> = Vec::new();
        output.resize((self.model_struct[self.model_struct.len()-1 as usize]).try_into().unwrap(), 0.0f64);
        
        //elementwise, and an offset for some of the derivatives later. (1-L*R), could also have made a seperate function to do this to the output, but this is simpler to me
        for e in 0..left.len() {
            output[e] = left[e] *  right[e] +offset;
        }
        output
    }


    pub fn scalarMultiplication( left: &Vec<f64>, right: f64 ) -> Vec<f64> {
        let mut output = Vec::with_capacity(left.len());

        for l in 0..left.len() {
            output[l] = left[l]*right;
        }

        output
    }

    pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> //borrowed from stack overflow
        where
        T: Clone,
        {
        assert!(!v.is_empty());
        (0..v[0].len())
        .map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
        .collect()
    }
    
    pub fn feed(&mut self, index:usize) -> f64{
        self.activations[0] = self.data.data[index].data.clone();


        //matrix abstraction ftw.
        for l in 1..self.activations.len() as usize{
            self.pre_activations[l] = MultilayerPerceptron::add(&MultilayerPerceptron::dot( &mut self.layers[l].weights, &mut self.activations[l-1]),  &self.layers[l].bias);
            self.activations[l] = MultilayerPerceptron::sigmoid(&self.pre_activations[l]);
            println!("Layer: {}, {:.4?}", l, self.activations[l][0]);
        }

        println!("pre_activations[{}]: {:.4?}", self.pre_activations.len()-1, self.pre_activations[self.pre_activations.len()-1]);
        println!("model[{}]: {:.4?}", self.activations.len()-1,  self.activations[self.activations.len()-1]); //should maybe also return result of feeding forward, for evaluation or visualization later.
        let mut target = &mut vec![0.0f64; 10];
        target[self.data.data[index].label as usize] = 1.0f64;
        return self.cost(&self.activations[self.activations.len()-1], target);
        
    }
    
    /*pub*/ fn backpropagate(&mut self, learning_rate: f64, epoch: f64, batching: BatchingType, index:usize) {
        
        for e in 0..epoch as usize {
            println!("Epoch: {}", e);
            //forward propagation
            match batching {
                BatchingType::MiniBatch(size) => {
                    /*
                    //We need structs like "weightsGradients" and "biasGradients" to contain this mess.
                    let mut weight_gradients: Vec<Vec<Vec<Vec<f64>>>> = Vec::new(); // there's no fucking way. array of each pass containing array of all layers, containing all neurons containing all weights.
                    let mut bias_gradients: Vec<Vec<Vec<f64>>> = Vec::new(); //array of each pass, containing every layer of that pass containing every neurons bias.
                    let mut costs: Vec<f64>/*Fresh air*/ = Vec::new();

                    for 0..epoch  :
                    for i in data  :
                    for l in layers  :
                        weight_gradient.push(  activations(a[l-1])*dsigmoid(pre_a)*error  );
                        bias_gradiant.push (  dsigmoid(pre_a)*error  );

                        if l%size==0 || l==activations.len()-1 //batch size reached, or end of training examples.
                            //average gradients
                            //backpropagate final gradients.

                    */

                }
                BatchingType::Batching => {//should we be counting the amount of bytes we enter into memory per pass, and force a backpropagation if there's not enough?  //is compression viable? speed-wise?
                    //same as miniBatch but just backpropagate after iterating over all data.
                }
                BatchingType::Online => {
                    /*
                    let cost = self.feed(index);
                    let mut error = cost;
                    for l in (1..self.activations.len()).rev() {

                        let weight_gradient = self.activations(&self.activations[l-1]) * dsigmoid(&self.pre_activations[l]) * error;
                        let bias_gradient = self.elementwise(&self.dsigmoid(&self.pre_activations[l]), &vec!(error), 0.0f64);

                        //Adjust weights+biases
                        &mut self.layers[l].weights = &mut self.layers[l].weights*(-weight_gradient)*learning_rate;
                        &mut self.layers[l].bias = &self.layers[l].bias*(-&bias_gradient)*&learning_rate; //we need a multiply function between &Vec<64> and &f64 or f64 it implements the clone trait, right?
                        error: f64 = /*self.activations[l-1] * dsigmoid(pre_activations)* */error //error of entire previous layer. //::dot?
                     */
                    }


                }
            }

            /* Pseudocode

            let cost = feed(index);
            let mut error = cost;
            for l in (1..activations.len()).rev() {
                dweights = inputs(a[l-1])*dsigmoid(pre_a)*error;
                dbias = dsigmoid(pre_a)*error;
                error: f64 = dz/da[l-1] * dsig(pre_activations)*Error //error of entire previous layer.
            }

            &mut self.layers[L].weights = MultilayerPerceptron::subtract(&mut self.layers[L].weights, MultilayerPerceptron::multiply(learning_rate, MultilayerPerceptron::dot(MultilayerPerceptron::transpose(activations(l-1)),Error))))


            */
        }

    }


