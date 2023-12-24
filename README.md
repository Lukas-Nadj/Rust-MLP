# AI::from("Scratch")

I remember failing at creating an ai from scratch for my school project, and i wanted to practice using rust, so 2 birds with one stone :)
some features i've thought about are options for precision and making the activation function a closure that you can import on init. so can define a simple network with just two vector arrays like

    let model_structure = vec![784, 16, 16, 10]
    let functions = vec![ReLu, ReLu, Sigmoid]

anyways i'm not creating anything thats gonna be too efficient or score that well. for now i'm just using sigmoids and creating a feed-forward neural network / Multi Layer Perceptron. 


## How to use it
This is mostly for fun and personal use, but if you want to use it you can create a model like:

    let mut model = MultilayerPerceptron::init( vec![10, 16, 16, 10]  ); // [input, hidden, hidden, output]
    model.feed();


## Todo
- Backpropagation / gradient descent
- import training data into memory. (using Mnist handwritten digits data set)
- mix n' match activation functions.
- vizualisation (might just stream data with a webserver to a local website and do that part in javascript)
    - visualise digits while training
    - live drawing fed to model.
- Evaluation
