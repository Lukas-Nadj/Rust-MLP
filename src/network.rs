
use rand::prelude::*;

#[derive(Debug)]
pub struct Layer {
    weights:  Vec<Vec<f32>>,
    bias: Vec<f32>,
}
#[derive(Debug)]
pub struct MultilayerPerceptron{
	pub activations: Vec<Vec<f32>>,
	pub layers: Vec<Layer>,
    pub model_struct: Vec<i32>
}

impl MultilayerPerceptron {

	pub fn init(  /*activation_function: F,*/ model_struct: Vec<i32> ) -> Self{	
        let mut activations: Vec<Vec<f32>> = vec![vec![]; model_struct.len()];
        
        let mut layers: Vec<Layer> = (0..model_struct.len()).map(|_| Layer {
            weights: vec![vec![]],
            bias: vec![],
        }).collect();
        
        //initialize activations array
        let mut rng = rand::thread_rng();
        for (i, size) in model_struct.iter().enumerate() { 
            activations[i].resize(*size as usize, 0.0f32); //insert closure that creates random values
        }
        
        for (l, size) in model_struct.iter().enumerate() { 
            layers[l].weights.resize(*size as usize, Vec::new());
            layers[l].bias.resize_with(*size as usize, || rng.gen::<f32>());
            for i in 0..*size as usize { 
                if(l>0){
                    layers[l].weights[i].resize_with(activations[l-1].len(), || rng.gen::<f32>());
                }
            }
        }
        
        MultilayerPerceptron { activations, layers, model_struct}
    }


    fn cost(activations: &Vec<f32>, target: &Vec<f32>) {
        let mut sum = 0.0f32;
        let mut cost: Vec<f32> = vec![0.0; target.len()];
        
        for (index, err) in activations.iter().enumerate() {
            cost[index] = (activations[index] - target[index]).powi(2);
            sum += (activations[index] - target[index]).powi(2);
        }
    }

    pub fn dot(left: &Vec<Vec<f32>>, right: &Vec<f32>) -> Vec<f32> {
        let mut output:Vec<f32> = Vec::with_capacity(left.len());
        let len = left.len();
        let len_inner = left[0].len();
        for neuron in 0..len {
            let mut sum = 0.0f32;
            
            for i in 0..len_inner {
                sum += left[neuron][i]*right[i];
            }
            
            output.push(sum);
        }
        output
    } 
    
    pub fn add(left: &Vec<f32>, right: &Vec<f32>) -> Vec<f32> {
        assert_eq!(left.len(), right.len());
        assert_ne!(left.len(),  0, "Fuck");
        let mut output:Vec<f32> = Vec::new();
        output.resize(left.len(), 0.0f32);
        
        if right.len() > 0 {
        for i in 0..right.len() {
            output[i] = left[i] + right[i];
        }
        }
        
        output
    }
    
    pub fn sigmoid(w_sum: &Vec<f32>) -> Vec<f32> {
        let mut output:Vec<f32> = Vec::new();
        output.resize(w_sum.len(), 0.0f32);
        
        for i in 0..w_sum.len() {
            output[i] = 1.0f32/(1.0f32+w_sum[i].exp());
        }
        
        output
    }
    
    pub fn elementwise(&self, left: &Vec<f32>, right: &Vec<f32>, offset: f32) -> Vec<f32> {
        let mut output: Vec<f32> = Vec::new();
        output.resize((self.model_struct[self.model_struct.len()-1 as usize]).try_into().unwrap(), 0.0f32);
        
        for e in 0..left.len() {
            output[e] = left[e] * (offset - right[e]);
        }
        output
    }
    
    pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>>
        where
        T: Clone,
        {
        assert!(!v.is_empty());
        (0..v[0].len())
        .map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
        .collect()
    }
    
    pub fn feed(&mut self) {
        
        for l in 1..self.activations.len() as usize{
            self.activations[l] = MultilayerPerceptron::sigmoid(&MultilayerPerceptron::add(&MultilayerPerceptron::dot( &mut self.layers[l].weights, &mut self.activations[l-1]),  &self.layers[l].bias));
            println!("model: {:?}", self.activations[l]);
        }
        
    }
    
    pub fn backpropagate(&mut self, learning_rate: f64) {
        /*
        let activation_error: Vec<Vec<f32>> = Vec::new();
        activation_error.resize(self.model_struct.length);
        let dW = dot(dot(cost(), self.activations[l]),  ) 
        
        
        probably wrong, but i'll figure it out when i actually write it.
         */
    }

}