/*********************************************************************
 * File  : PerceptronMulticapa.cpp
 * Date  : 2017
 *********************************************************************/

#include "PerceptronMulticapa.h"
#include "util.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <limits>
#include <math.h>

#include <random>


using namespace imc;
using namespace std;
using namespace util;

double randomNumber(double min, double max){
	return min + (double)rand()/RAND_MAX *(max-min);
}

// ------------------------------
// CONSTRUCTOR: Dar valor por defecto a todos los parámetros
PerceptronMulticapa::PerceptronMulticapa(){
	dDecremento = 1;
	dValidacion = 0.0;
	dEta = 0.1;
	dMu = 0.9;
	nNumCapas = 1;
	pCapas = NULL;
}

// ------------------------------
// Reservar memoria para las estructuras de datos
int PerceptronMulticapa::inicializar(int nl, int npl[]) {
	nNumCapas = nl;
	pCapas = new Capa[nl];
	for (int i = 0; i < nl; i++)
	{
		pCapas[i].nNumNeuronas = npl[i];
		pCapas[i].pNeuronas = new Neurona[npl[i]];
	}
	
	return 1;
}

// ------------------------------
// DESTRUCTOR: liberar memoria
PerceptronMulticapa::~PerceptronMulticapa() {
	liberarMemoria();
}

// ------------------------------
// Liberar memoria para las estructuras de datos
void PerceptronMulticapa::liberarMemoria() {
	for (int i = 0; i < nNumCapas; i++)
	{
		delete[] pCapas[i].pNeuronas;
	}
	delete[] pCapas;
}

// ------------------------------
// Rellenar todos los pesos (w) aleatoriamente entre -1 y 1
void PerceptronMulticapa::pesosAleatorios() {
	// Recorremos todas las capas menos la de entrada
	for (int i = 1; i < nNumCapas; i++)
	{
		// Recorremos todas las neuronas de la capa
		for (int j = 0; j < pCapas[i].nNumNeuronas; j++)
		{
			// Calculamos el numero de neuronas de la capa anterior junto con el sesgo
			int number_weights = pCapas[i-1].nNumNeuronas+1;

			// Reserva memoria para las variables de las neuronas
			pCapas[i].pNeuronas[j].w = new double[number_weights];
			pCapas[i].pNeuronas[j].deltaW = new double[number_weights];
			pCapas[i].pNeuronas[j].ultimoDeltaW = new double[number_weights];
			pCapas[i].pNeuronas[j].wCopia = new double[number_weights];

			// Inicializamos los pesos y las variables adicionales de las neuronas
			for (int k = 0; k < number_weights; k++)
			{
				pCapas[i].pNeuronas[j].w[k] = randomNumber(-1,1);
				pCapas[i].pNeuronas[j].deltaW[k] = 0.0;
				pCapas[i].pNeuronas[j].ultimoDeltaW[k] = 0.0;
				pCapas[i].pNeuronas[j].wCopia[k] = 0.0;
			}
		}
	}
}

// ------------------------------
// Alimentar las neuronas de entrada de la red con un patrón pasado como argumento
void PerceptronMulticapa::alimentarEntradas(double* input) {
	for (int i = 0; i < pCapas[0].nNumNeuronas; i++)
	{
		pCapas[0].pNeuronas[i].x = input[i];// La salida de la primera capa es la entrada de la red
	}
}

// ------------------------------
// Recoger los valores predichos por la red (out de la capa de salida) y almacenarlos en el vector pasado como argumento
void PerceptronMulticapa::recogerSalidas(double* output){
	int number_output = pCapas[nNumCapas-1].nNumNeuronas; //nNumCapas-1 porque empieza en 0
	for (int i = 0; i < number_output; i++)
	{
		output[i] = pCapas[nNumCapas-1].pNeuronas[i].x;// Guardamos la salida de la ultima capa oculta en el vector de salida
	}
}

// ------------------------------
// Hacer una copia de todos los pesos (copiar w en copiaW)
void PerceptronMulticapa::copiarPesos() {
	// Recorremos todas las capas
	for (int i = 1; i < nNumCapas; i++)
	{
		// Recorremos todas las neuronas
		for (int j = 0; j < pCapas[i].nNumNeuronas; j++)
		{
			// Calculamos el número de neuronas de la capa anterior junto con el sesgo (por eso el +1)
			int number_weights = pCapas[i-1].nNumNeuronas+1;
			// Restauramos los pesos
			for (int k = 0; k < number_weights; k++)
			{
				pCapas[i].pNeuronas[j].wCopia[k] = pCapas[i].pNeuronas[j].w[k];
			}
		}
	}
}

// ------------------------------
// Restaurar una copia de todos los pesos (copiar copiaW en w)
void PerceptronMulticapa::restaurarPesos() {
	// Recorremos todas las capas
	for (int i = 1; i < nNumCapas; i++)
	{
		// Recorremos todas las neuronas
		for (int j = 0; j < pCapas[i].nNumNeuronas; j++)
		{
			// Calculamos el número de neuronas de la capa anterior junto con el sesgo (por eso el +1)
			int number_weights = pCapas[i-1].nNumNeuronas+1;
			// Restauramos los pesos
			for (int k = 0; k < number_weights; k++)
			{
				pCapas[i].pNeuronas[j].w[k] = pCapas[i].pNeuronas[j].wCopia[k];
			}
		}
	}
}

// ------------------------------
// Calcular y propagar las salidas de las neuronas, desde la primera capa hasta la última
void PerceptronMulticapa::propagarEntradas() {
	// Leemos todas las capas menos la de entrada (Empezamos en 1 porque no se cuenta la capa de entrada)
	for (int i = 1; i < nNumCapas; i++)
	{
		// Leemos las neuronas de la capa
		for (int j = 0; j < pCapas[i].nNumNeuronas; j++)
		{
			double activation=0.0;

			// Numero de neuronas de la capa anterior
			int number_weights = pCapas[i-1].nNumNeuronas;
			// Calculamos sum(w^(capaActual) * out^(CapaAnterior))			
			for (int k = 0; k < number_weights; k++)
			{
				activation = activation + (pCapas[i].pNeuronas[j].w[k]) * (pCapas[i-1].pNeuronas[k].x);
			}
			// Aplicamos el sesgo de esa neurona
			activation = activation + pCapas[i].pNeuronas[j].w[number_weights];
			// Aplicamos sigmoide
			pCapas[i].pNeuronas[j].x = 1/(1+exp(-activation));
		}
	}
}

// ------------------------------
// Calcular el error de salida (MSE) del out de la capa de salida con respecto a un vector objetivo y devolverlo
double PerceptronMulticapa::calcularErrorSalida(double* target) {
	double MSE = 0.0;
	int number_output = pCapas[nNumCapas-1].nNumNeuronas;

	for (int i = 0; i < number_output; i++)
	{
		MSE = MSE + pow((pCapas[nNumCapas-1].pNeuronas[i].x - target[i]),2);
	}
	MSE = MSE/number_output;
	return MSE;
}

// ------------------------------
// Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
void PerceptronMulticapa::retropropagarError(double* objetivo) {
	// Leemos las neuronas de la capa de salida
	for (int i = 0; i < pCapas[nNumCapas-1].nNumNeuronas; i++)
	{
		// Calculamos la derivada de cada una de las neuronas de la capa de salida
		pCapas[nNumCapas-1].pNeuronas[i].dX = -(objetivo[i] - pCapas[nNumCapas-1].pNeuronas[i].x) * pCapas[nNumCapas-1].pNeuronas[i].x * (1 - pCapas[nNumCapas-1].pNeuronas[i].x);
	}

	// numero_capas-2 porque queremos la ultima capa oculta, es decir la penúltima capa
	for (int i = nNumCapas-2; i > 0; i--)
	{
		for (int j = 0; j < pCapas[i].nNumNeuronas; j++)
		{
			double aux = 0;
			// Todas las neuronas de la capa siguiente conectadas con la neurona j
			for (int k = 0; k < pCapas[i+1].nNumNeuronas; k++)
			{
				// w*delta (Sumatorio)
				aux = aux + pCapas[i+1].pNeuronas[k].w[j] * pCapas[i+1].pNeuronas[k].dX;
			}
			pCapas[i].pNeuronas[j].dX = aux * pCapas[i].pNeuronas[j].x * (1 - pCapas[i].pNeuronas[j].x);// Derivada
		}
	}
}

// ------------------------------
// Acumular los cambios producidos por un patrón en deltaW
void PerceptronMulticapa::acumularCambio() {
	// Recorrer todas las capas
	for (int i = 1; i < nNumCapas; i++)
	{
		// Recorrer todas las neuronas
		for (int j = 0; j < pCapas[i].nNumNeuronas; j++)
		{
			// Recorrer todas las neuronas de la capa anterior
			for (int k = 0; k < pCapas[i-1].nNumNeuronas; k++)
			{
				pCapas[i].pNeuronas[j].deltaW[k] = pCapas[i].pNeuronas[j].deltaW[k] + pCapas[i].pNeuronas[j].dX * pCapas[i-1].pNeuronas[k].x;	
			}
			// Hay que tener en cuenta el sesgo
			pCapas[i].pNeuronas[j].deltaW[pCapas[i-1].nNumNeuronas] = pCapas[i].pNeuronas[j].deltaW[pCapas[i-1].nNumNeuronas] + pCapas[i].pNeuronas[j].dX;
		}
	}
}

// ------------------------------
// Actualizar los pesos de la red, desde la primera capa hasta la última
void PerceptronMulticapa::ajustarPesos() {
	double new_eta = 0.0;
	// Recorremos las capas
	for (int i = 1; i < nNumCapas; i++)
	{
		// Decremento de learning rate según al capa
		new_eta = pow(dDecremento, -(nNumCapas-i))*dEta;
		// Recorremos todas las neuronas de las capas
		for (int j = 0; j < pCapas[i].nNumNeuronas; j++)
		{
			// Recorremos todas las neuronas de la capa anterior
			for (int k = 0; k < pCapas[i-1].nNumNeuronas; k++)
			{
				// Ajustamos los pesos de la capa actual
				pCapas[i].pNeuronas[j].w[k] = pCapas[i].pNeuronas[j].w[k] - (new_eta * pCapas[i].pNeuronas[j].deltaW[k]) - (dMu *(new_eta * pCapas[i].pNeuronas[j].ultimoDeltaW[k])); 
				// Guardamos los cambios aplicados al peso
				pCapas[i].pNeuronas[j].ultimoDeltaW[k] = pCapas[i].pNeuronas[j].deltaW[k];
				// Reinicializamos deltaW ya que es online
				pCapas[i].pNeuronas[j].deltaW[k] = 0.0;
			}
			// Ajustamos el sesgo
			pCapas[i].pNeuronas[j].w[pCapas[i-1].nNumNeuronas] = pCapas[i].pNeuronas[j].w[pCapas[i-1].nNumNeuronas] - (new_eta * pCapas[i].pNeuronas[j].deltaW[pCapas[i-1].nNumNeuronas]) - (dMu * (new_eta * pCapas[i].pNeuronas[j].ultimoDeltaW[pCapas[i-1].nNumNeuronas]));
			pCapas[i].pNeuronas[j].ultimoDeltaW[pCapas[i-1].nNumNeuronas] = pCapas[i].pNeuronas[j].deltaW[pCapas[i-1].nNumNeuronas];
			pCapas[i].pNeuronas[j].deltaW[pCapas[i-1].nNumNeuronas] = 0.0;
		}
	}
}

// ------------------------------
// Imprimir la red, es decir, todas las matrices de pesos
void PerceptronMulticapa::imprimirRed() {
	// Recorremos todas las capas menos la de entrada
	int number_weights;
	for (int i = 1; i < nNumCapas; i++)
	{
		cout<<"\tLayer: "<<i<<"\n\t--------"<<endl;
		// Capa anterior con el sesgo aplicado
		number_weights = pCapas[i-1].nNumNeuronas+1;
		for (int j = 0; j < pCapas[i].nNumNeuronas; j++)
		{
			for (int k = 0; k < number_weights; k++)
			{
				cout<<pCapas[i].pNeuronas[j].w[k]<<" ";
			}
			cout<<endl;
		}
	}
}

// ------------------------------
// Simular la red: propagar las entradas hacia delante, retropropagar el error y ajustar los pesos
// entrada es el vector de entradas del patrón y objetivo es el vector de salidas deseadas del patrón
void PerceptronMulticapa::simularRedOnline(double* entrada, double* objetivo) {
	// Primero alimentamos las entradas
	alimentarEntradas(entrada);
	// Propagamos la entrada por toda la red
	propagarEntradas();
	// Retropropagamos el error desde la capa de salida hasta la de entrada
	retropropagarError(objetivo);
	// Acumulamos los cambios hechos
	acumularCambio();
	// Ajustamos los antiguos pesos para disminuir el error
	ajustarPesos();
}

// ------------------------------
// Leer una matriz de datos a partir de un nombre de fichero y devolverla
Datos* PerceptronMulticapa::leerDatos(const char *archivo) {
	Datos *network = new Datos;
	ifstream file(archivo);

	file >> network->nNumEntradas>>network->nNumSalidas>>network->nNumPatrones;

	// Reservamos memoria para las entradas y salidas
	network->entradas = new double*[network->nNumPatrones];
	network->salidas = new double*[network->nNumPatrones];

	for (int i = 0; i < network->nNumPatrones; i++)
	{
		network->entradas[i] = new double[network->nNumEntradas];
		network->salidas[i] = new double[network->nNumSalidas];
		for (int j = 0; j < network->nNumEntradas; j++)
		{
			file>>network->entradas[i][j];
		}
		for (int j = 0; j < network->nNumSalidas; j++)
		{
			file>>network->salidas[i][j];
		}
	}
	file.close();
	return network;
}

// ------------------------------
// Entrenar la red on-line para un determinado fichero de datos
void PerceptronMulticapa::entrenarOnline(Datos* pDatosTrain) {
	int i;
	for (i = 0; i < pDatosTrain->nNumPatrones; i++)
	{
		simularRedOnline(pDatosTrain->entradas[i], pDatosTrain->salidas[i]);
	}		
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el error MSE cometido
double PerceptronMulticapa::test(Datos* pDatosTest) {
	double MSE = 0;
	for (int i = 0; i < pDatosTest->nNumPatrones; i++)
	{
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		MSE = MSE + calcularErrorSalida(pDatosTest->salidas[i]);
	}
	MSE = MSE/(pDatosTest->nNumPatrones);
	return MSE;
}

// ------------------------------
// Ejecutar el algoritmo de entrenamiento durante un número de iteraciones, utilizando pDatosTrain
// Una vez terminado, probar como funciona la red en pDatosTest
// Tanto el error MSE de entrenamiento como el error MSE de test debe calcularse y almacenarse en errorTrain y errorTest
void PerceptronMulticapa::ejecutarAlgoritmoOnline(Datos * pDatosTrain, Datos * pDatosTest, int maxiter, double *errorTrain, double *errorTest){
	int countTrain = 0;

	// Inicialización de pesos
	pesosAleatorios();

	double minTrainError = 0.0;
	double minimum_validation_error = 0.0;

	int numSinMejorar;
	int validation_sinMejorar = 0.0;// Contador sin mejorar para validacion

	double testError = 0.0;	
	double validationError=0.0;
	
	int *permutacion = NULL;
	
	Datos *pDatosValidation = new Datos;// Para guardar los datos de validacion
	Datos *pDatosTrain_copia = new Datos;	

	// Generar datos de validación
	if(dValidacion > 0 && dValidacion < 1){
		
		// Iniciar datos para la validacion
		pDatosValidation->nNumPatrones = int(pDatosTrain->nNumPatrones * dValidacion); //dValidacion para coger un porcentaje de train
		pDatosValidation->nNumEntradas = pDatosTrain->nNumEntradas;
		pDatosValidation->nNumSalidas = pDatosTrain->nNumSalidas;

		// Guardamos los datos de train en la variable copia creada
		pDatosTrain_copia->nNumPatrones = pDatosTrain->nNumPatrones - int(pDatosTrain->nNumPatrones * dValidacion);
		pDatosTrain_copia->nNumEntradas = pDatosTrain->nNumEntradas;
		pDatosTrain_copia->nNumSalidas = pDatosTrain->nNumSalidas;

		// Reservamos memoria para la matriz de los datos, tanto para la validación como para la copia de train
		pDatosTrain_copia->salidas = new double*[pDatosTrain_copia->nNumPatrones];
		pDatosTrain_copia->entradas = new double*[pDatosTrain_copia->nNumPatrones];

		pDatosValidation->entradas = new double*[pDatosValidation->nNumPatrones];
		pDatosValidation->salidas = new double*[pDatosValidation->nNumPatrones];
		

		for (int i = 0; i < pDatosValidation->nNumPatrones; i++)
		{
			pDatosValidation->entradas[i] = new double[pDatosValidation->nNumEntradas];
			pDatosValidation->salidas[i] = new double[pDatosValidation->nNumSalidas];
		}

		for (int j = 0; j < pDatosTrain_copia->nNumPatrones; j++)
		{
			pDatosTrain_copia->entradas[j] = new double[pDatosTrain_copia->nNumEntradas];
			pDatosTrain_copia->salidas[j] = new double[pDatosTrain_copia->nNumSalidas];
		}

		// Inicializamos el vector con indices aleatorios para conseguir la validacion
		permutacion = vectorAleatoriosEnterosSinRepeticion(0, pDatosTrain->nNumPatrones-1, pDatosValidation->nNumPatrones);
		
		
		int validation_index = 0;
		bool isValidation = false;

		// Dividimos las permutaciones en dos vectores, un nuevo entrenamiento y uno de validacion
		for (int i = 0; i < pDatosTrain->nNumPatrones; i++)
		{
			isValidation = false;
			for (int j = 0; j < pDatosValidation->nNumPatrones; j++)
			{
				if (permutacion[j] == i)
				{
					isValidation = true;
				}
			}

			if (isValidation == true)
			{
				for (int k = 0; k < pDatosValidation->nNumEntradas; k++)
				{
					pDatosValidation->entradas[validation_index][k] = pDatosTrain->entradas[i][k];
				}
				for (int k = 0; k < pDatosValidation->nNumSalidas; k++)
				{
					pDatosValidation->salidas[validation_index][k] = pDatosTrain->entradas[i][k];
				}
				validation_index++;
			}
			else{
				for (int k = 0; k < pDatosTrain_copia->nNumEntradas; k++)
				{
					pDatosTrain_copia->entradas[validation_index][k] = pDatosTrain->entradas[i][k];
				}
				for (int k = 0; k < pDatosTrain_copia->nNumSalidas; k++)
				{
					pDatosTrain_copia->salidas[validation_index][k] = pDatosTrain->entradas[i][k];
				}
			}
		}
		validationError = test(pDatosValidation);
	}else{ // Si no hay validacion
		pDatosTrain_copia = pDatosTrain;
	}


	ofstream file("errores.txt");// Fichero para guardar los errores cometidos en cada iteracion
	// Aprendizaje del algoritmo
	do {

		entrenarOnline(pDatosTrain);
		double trainError = test(pDatosTrain);
		testError = test(pDatosTest);
		if( countTrain == 0 || fabs(trainError - minTrainError) > 0.00001){
			minTrainError = trainError;
			copiarPesos();
			numSinMejorar=0;
		}else{
			numSinMejorar++;
		}

		if(numSinMejorar==50){
			cout << "Salida porque no mejora el entrenamiento!!"<< endl;
			restaurarPesos();
			countTrain = maxiter;
		}

		countTrain++;
		// Comprobamos condiciones
		if (dValidacion>0 && dValidacion<1)
		{
			validationError = test(pDatosValidation);
			if(countTrain == 0 || fabs(validationError - minimum_validation_error) > 0.00001){
				minimum_validation_error = validationError;
				copiarPesos();
				validation_sinMejorar = 0;
			}else{
				validation_sinMejorar++;
			}
			if(validation_sinMejorar == 50){
				restaurarPesos();
				countTrain = maxiter;
			}
			cout << "Iteración " << countTrain << "\t Error de entrenamiento: " << trainError <<"\t Error de test: "<<testError<< "\t Error de validación: " << validationError << endl;
			file << countTrain<<","<< trainError<<","<<testError<<endl;
		}else{
			cout << "Iteración " << countTrain << "\t Error de entrenamiento: " << trainError << "\t Error de test: "<<testError<<endl;
			file << countTrain<<","<< trainError<<","<<testError<<endl;
		}
		
		// Comprobar condiciones de parada de validación y forzar
		// OJO: en este caso debemos guardar el error de validación anterior, no el mínimo
		// Por lo demás, la forma en que se debe comprobar la condición de parada es similar
		// a la que se ha aplicado más arriba para el error de entrenamiento

	} while ( countTrain<maxiter );
	file.close();

	cout << "PESOS DE LA RED" << endl;
	cout << "===============" << endl;
	imprimirRed();

	cout << "Salida Esperada Vs Salida Obtenida (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<pDatosTest->nNumPatrones; i++){
		double* prediccion = new double[pDatosTest->nNumSalidas];

		// Cargamos las entradas y propagamos el valor
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		recogerSalidas(prediccion);
		for(int j=0; j<pDatosTest->nNumSalidas; j++)
			cout << pDatosTest->salidas[i][j] << " -- " << prediccion[j] << " ";
		cout << endl;
		delete[] prediccion;

	}

	testError = test(pDatosTest);
	*errorTest=testError;
	*errorTrain=minTrainError;
	file.close();
}

