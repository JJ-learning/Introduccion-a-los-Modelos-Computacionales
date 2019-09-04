# How to run this program
### First, go to P2/practica2. In order to compile it, run the following command:
> g++ practica2.cpp imc/PerceptronMulticapa.cpp imc/util.cpp -std=c++11 -O3 -o practica2
### Then, in order to run the program, do as follows:
> ./practica2 -[arguments]
### Type of arguments:
1. -T: File for the test file.
2. -t: File with the training params.
3. -i: Number of iterations.
4. -l: Number of hidden layers.
5. -h: Number of neurons on each layer.
6. -e: Value for the eta param.
7. -m: Value for the mu param.
8. -v: Value for the training ratio.
9. -d: Value for the decrement.
10. -o: Mode online or offline. By default: Offline
11. -f: Mode of error. 0-> MSE 1-> Cross entropy
12. -s: Mode of activation function. By default Sigmoid, if activated Softmax
