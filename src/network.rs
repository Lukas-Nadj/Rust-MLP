
use rand::prelude::*;
use crate::data::Data; 


#[derive(Debug)]
pub struct Layer { //:3
    weights:  Vec<Vec<f64>>,
    bias: Vec<f64>,
}
#[derive(Debug)]
pub struct MultilayerPerceptron<'a>{
    pub data: &'a Data,
	pub activations: Vec<Vec<f64>>,
	pub layers: Vec<Layer>, //could become a vec array of touples, or be defined inline without Layer struct. but for now that would be a headache.
    pub model_struct: Vec<i32> //could simplify some loops later
}

pub enum BatchingType { 
    MiniBatch(u32),
    Batching,
    Online
}

impl<'a> MultilayerPerceptron<'a> {
    pub fn load( model_fs:i32 , data:&'a Data ) /* -> Self */{
        
    }
}

impl<'a> MultilayerPerceptron<'a> {
    
	pub fn init( model_struct: Vec<i32>, data:&'a Data ) -> Self{	 //TODO: assert data.data has the correct dimensions. with input layer.
        let mut activations: Vec<Vec<f64>> = vec![vec![]; model_struct.len()]; //initialize array size to model size
        
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
            layers[l].weights.resize(*size as usize, Vec::new());
            layers[l].bias.resize_with(*size as usize, || rng.gen::<f64>());
            for i in 0..*size as usize { 
                if(l>0){
                    //i fucking love closures and resize_with !!!
                    layers[l].weights[i].resize_with(activations[l-1].len(), || rng.gen::<f64>());  //insert closure that creates random values
                }
            }
        }
        //return a new model struct that takes ownership of these variables. //nope, we should just use the Self struct instead. now we can init an already initialized model without creating a new struct? a little useless.
        Self {data, activations, layers, model_struct}
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
    
    pub fn sigmoid(z: &Vec<f64>) -> Vec<f64> { //applies activation function to an entire layer.
        //init
        let mut output:Vec<f64> = Vec::new();
        output.resize(z.len(), 0.0f64);
        
        //Sigmoid activation
        for i in 0..z.len() {
            output[i] = 1.0f64/(1.0f64+z[i].exp());
        }
        
        output
    }
    
    pub fn elementwise(&self, left: &Vec<f64>, right: &Vec<f64>, offset: f64) -> Vec<f64> {
        //init
        let mut output: Vec<f64> = Vec::new();
        output.resize((self.model_struct[self.model_struct.len()-1 as usize]).try_into().unwrap(), 0.0f64);
        
        //elementwise, and an offset for some of the derivatives later. (1-L*R), could also have made a seperate function to do this to the output, but this is simpler to me
        for e in 0..left.len() {
            output[e] = left[e] * (offset - right[e]);
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
    
    pub fn feed(&mut self) -> f64{
        
        //matrix abstraction ftw.
        for l in 1..self.activations.len() as usize{
            // Z = σ(W*A[l-1]+B), basically takes the sigmoid of (weighted sum's + biases)
            self.activations[l] = MultilayerPerceptron::sigmoid(&MultilayerPerceptron::add(&MultilayerPerceptron::dot( &mut self.layers[l].weights, &mut self.activations[l-1]),  &self.layers[l].bias));
            
        }
        
        println!("model: {:.4?}", self.activations[self.activations.len()-1]); //should maybe also return result of feeding forward, for evaluation or visualization later.
        return self.cost(&self.activations[self.activations.len()-1], &vec![0.0f64, 1.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64]);
        
    }
    
    pub fn backpropagate(&mut self, learning_rate: f64, epoch: f64, batching: BatchingType) {
        
        let error:Vec<f64> = Vec::new();
        let batchsize:i32 = 1;
        let cost: f64;
        for e in 0..epoch as usize {
            println!("Epoch: {}", e);
            //forward propagation
            match batching {
                BatchingType::MiniBatch(size) => {
                    //error.resize_with(self.data.data.len(), f)
                    //iterate over slice with `size` iterate over each
                }
                BatchingType::Batching => {
                    // batchsize = &mut self.data.data.len();
                    /* 
                    
                    let mut sum:f64 = 0.0f64;
                    
                    for err in &feed result { 
                    sum += err;
                    }
                    
                    cost = sum/error.len() as f64;
                    
                    
                    */
                }
                
                _ => {
                    
                }
            }
        
            //final cost
            
        
            //backpropagation
            /*
            for cost in 0..erro.iter.len().whatever { //iterate over each batch
            let mut prevlayer:f64 = cost.clone(); //variable for dC/da(l-1), cost with respect to the previous layer
            for i in &mut self.data.data.len()..0 {
                //weights -=
                //bias -=
                //prevlayer =
            }
            
             */
            
        }
        
        
        /*
        let activation_error: Vec<Vec<f32>> = Vec::new();
        activation_error.resize(self.model_struct.length);
        let dW = dot(dot(cost(), self.activations[l]),  ) 
        
        
        probably wrong, but i'll figure it out when i actually try to write it.
         */
    }

}