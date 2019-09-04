/*********************************************************************
 * File  : PerceptronMulticapa.cpp
 * Date  : 2018
 *********************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <limits>
#include <math.h>

#include "PerceptronMulticapa.h"
#include "util.h"

using namespace imc;
using namespace std;
using namespace util;

double randomNumber(double min, double max){
	return min + (double)rand()/RAND_MAX *(max-min);
}

// ------------------------------
// CONSTRUCTOR: Dar valor por defecto a todos los parámetros (dEta, dMu, dValidacion y dDecremento)
PerceptronMulticapa::PerceptronMulticapa(){
	dDecremento = 1;
	dValidacion = 0.0;
	dEta = 0.1;
	dMu = 0.9;
	nNumCapas = 1;
	pCapas = NULL;
	nNumPatronesTrain = 0;
	bOnline = false; 
}

// ------------------------------
// Reservar memoria para las estructuras de datos
// nl tiene el numero de capas y npl es un vector que contiene el número de neuronas por cada una de las capas
// tipo contiene el tipo de cada capa (0 => sigmoide, 1 => softmax)
// Rellenar vector Capa* pCapas
int PerceptronMulticapa::inicializar(int nl, int npl[], bool tipo) {
	nNumCapas = nl;
	pCapas = new Capa[nl];
	for (int i = 0; i < nl; i++)
	{
		pCapas[i].nNumNeuronas = npl[i];
		pCapas[i].pNeuronas = new Neurona[npl[i]];
	}

	if(tipo == true){// Capa de salida tipo softmax
		pCapas[nNumCapas-1].tipo = 1;
	}else{// Capa de salida tipo sigmoid
		pCapas[nNumCapas-1].tipo = 0;
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
	//Output of the softmax function
	double outputSum = 0.0;
	// Leemos todas las capas
	for (int i = 1; i < nNumCapas; ++i)
	{
		// Leemos todas las neuronas de la capa
		for (int j = 0; j < pCapas[i].nNumNeuronas; ++j)
		{
			double activation = 0.0;
			// Numero de neuronas de la capa anterior
			int number_weights = pCapas[i-1].nNumNeuronas;
			for (int k = 0; k < number_weights; ++k)
			{
				activation += (pCapas[i].pNeuronas[j].w[k] * pCapas[i-1].pNeuronas[k].x);
			}
			// Aplicamos el sesgo a esa neurona
			activation += (pCapas[i].pNeuronas[j].w[number_weights]);

			// Si la capa de salida es softmax
			if(i == (nNumCapas-1) && pCapas[nNumCapas-1].tipo == 1){
				pCapas[i].pNeuronas[j].x = exp(activation);
				outputSum += pCapas[i].pNeuronas[j].x;
			}else{
				pCapas[i].pNeuronas[j].x = 1/(1+exp(-activation));
			}
		}
	}
	// Si estamos usando la funcion softmax, normalizamos
	if(pCapas[nNumCapas-1].tipo == 1){
		// Recorremos todas las neuronas de la capa de salida
		for (int i = 0; i < pCapas[nNumCapas-1].nNumNeuronas; ++i)
		{
			pCapas[nNumCapas-1].pNeuronas[i].x /= outputSum;
		}
	}
}

// ------------------------------
// Calcular el error de salida del out de la capa de salida con respecto a un vector objetivo y devolverlo
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
double PerceptronMulticapa::calcularErrorSalida(double* target, int funcionError) {
	double tipoError = 0.0;
	int number_output = pCapas[nNumCapas-1].nNumNeuronas;

	if (funcionError == 0)// Error tipo MSE
	{
		// Recorremos todas las neuronas de la capa de salida
		for (int i = 0; i < number_output; i++)
		{
			tipoError += pow((pCapas[nNumCapas-1].pNeuronas[i].x - target[i]),2);
		}
		tipoError /= number_output;
	}else{// Error tipo entropia cruzada
		// Recorremos todas las neuronas de la capa de salida
		for (int i = 0; i < number_output; ++i)
		{
			if (pCapas[nNumCapas-1].pNeuronas[i].x != 0)
			{
				tipoError -= (target[i] * log(pCapas[nNumCapas-1].pNeuronas[i].x));
			}
		}
		tipoError /= number_output;
	}
	return tipoError;
}

// ------------------------------
// Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void PerceptronMulticapa::retropropagarError(double* objetivo, int funcionError) {
	// Tipo sigmoide
	if (pCapas[nNumCapas-1].tipo == 0)
	{
		for (int i = 0; i < pCapas[nNumCapas-1].nNumNeuronas; ++i)
		{
			if (funcionError == 0)// Tipo MSE
			{
				// Calculamos la derivada de cada una de las neuronas de la capa de salida
				pCapas[nNumCapas-1].pNeuronas[i].dX = -(objetivo[i] - pCapas[nNumCapas-1].pNeuronas[i].x) * pCapas[nNumCapas-1].pNeuronas[i].x * (1 - pCapas[nNumCapas-1].pNeuronas[i].x);
			}else// Tipo entropia cruzada
			{
				// Calculamos la derivada de cada una de las neuronas de la capa de salida
				pCapas[nNumCapas-1].pNeuronas[i].dX = -(objetivo[i] / pCapas[nNumCapas-1].pNeuronas[i].x) * pCapas[nNumCapas-1].pNeuronas[i].x * (1 - pCapas[nNumCapas-1].pNeuronas[i].x);
			}
		}
	}else{// Tipo softmax
		// Recorremos las neuronas de la capa de salida
		for (int i = 0; i < pCapas[nNumCapas-1].nNumNeuronas; ++i)
		{
			double outputSum = 0.0;
			for (int j = 0; j < pCapas[nNumCapas-1].nNumNeuronas; ++j)
			{
				if(funcionError == 0){
					// Ponemos (j==i) para que a 1 le reste la salida o para que se haga negativo
					outputSum -= (objetivo[j] - pCapas[nNumCapas-1].pNeuronas[j].x) * pCapas[nNumCapas-1].pNeuronas[i].x * ((j==i) - pCapas[nNumCapas-1].pNeuronas[j].x);
				}else{
					outputSum -= (objetivo[j] / pCapas[nNumCapas-1].pNeuronas[j].x) * pCapas[nNumCapas-1].pNeuronas[i].x * ((j==i) - pCapas[nNumCapas-1].pNeuronas[j].x);
				}
			}
			pCapas[nNumCapas-1].pNeuronas[i].dX = outputSum;
		}
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
				aux += pCapas[i+1].pNeuronas[k].w[j] * pCapas[i+1].pNeuronas[k].dX;
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
				// Reinicializamos deltaW
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
// Simular la red: propragar las entradas hacia delante, computar el error, retropropagar el error y ajustar los pesos
// entrada es el vector de entradas del patrón, objetivo es el vector de salidas deseadas del patrón.
// El paso de ajustar pesos solo deberá hacerse si el algoritmo es on-line
// Si no lo es, el ajuste de pesos hay que hacerlo en la función "entrenar"
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void PerceptronMulticapa::simularRed(double* entrada, double* objetivo, int funcionError) {
	// Primero alimentamos las entradas
	alimentarEntradas(entrada);
	// Propagamos la entrada por toda la red
	propagarEntradas();
	// Retropropagamos el error desde la capa de salida hasta la de entrada
	retropropagarError(objetivo, funcionError);
	// Acumulamos los cambios hechos
	acumularCambio();
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
	// Rellenamos las entradas y las salidas
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
// Entrenar la red para un determinado fichero de datos (pasar una vez por todos los patrones)
void PerceptronMulticapa::entrenar(Datos* pDatosTrain, int funcionError) {
	// Si es off-line
	if(bOnline == false){
		for (int i = 0; i < pDatosTrain->nNumPatrones; i++)
		{
			simularRed(pDatosTrain->entradas[i], pDatosTrain->salidas[i], funcionError);
		}	
		// Ajustamos los pesos para todos los patrones de una vez
		ajustarPesos();
	}else{// Si es on-line
		for (int i = 0; i < pDatosTrain->nNumPatrones; i++)
		{
			simularRed(pDatosTrain->entradas[i], pDatosTrain->salidas[i], funcionError);
			// Ajustamos los pesos para cada patron
			ajustarPesos();
		}		
	}
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el error cometido
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
double PerceptronMulticapa::test(Datos* pDatosTest, int funcionError) {
	double tipoError = 0;
	for (int i = 0; i < pDatosTest->nNumPatrones; i++)
	{
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		tipoError += calcularErrorSalida(pDatosTest->salidas[i], funcionError);
	}
	if (funcionError == 0)// MSE
	{
		tipoError /= (pDatosTest->nNumPatrones);
	}else{// Entropia cruzada
		tipoError /= (pDatosTest->nNumPatrones * pCapas[nNumCapas-1].nNumNeuronas);
	}
		
	return tipoError;
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el CCR
double PerceptronMulticapa::testClassification(Datos* pDatosTest) {
	double ErrorCCR = 0.0;
	// Leemos todos los patrones
	for (int i = 0; i < pDatosTest->nNumPatrones; ++i)
	{
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();

		int target = 0;
		int output = 0;
		double maxDP = 0.0;// Error cometido clase incorrecta
		double maxOP = 0.0;// Error cometido clase correcta

		// Leemos la capa de salida
		for (int j = 0; j < pCapas[nNumCapas-1].nNumNeuronas; ++j)
		{
			if (pDatosTest->salidas[i][j] > maxDP)
			{
				maxDP = pDatosTest->salidas[i][j];
				target = j;
			}
			if(pCapas[nNumCapas-1].pNeuronas[j].x > maxOP){
				maxOP = pCapas[nNumCapas-1].pNeuronas[j].x;
				output = j;
			}
		}

		if(target == output){
			ErrorCCR++;
		}
	}
	ErrorCCR = 100 * (ErrorCCR/pDatosTest->nNumPatrones);

	return ErrorCCR;
}

// ------------------------------
// Ejecutar el algoritmo de entrenamiento durante un número de iteraciones, utilizando pDatosTrain
// Una vez terminado, probar como funciona la red en pDatosTest
// Tanto el error MSE de entrenamiento como el error MSE de test debe calcularse y almacenarse en errorTrain y errorTest
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void PerceptronMulticapa::ejecutarAlgoritmo(Datos * pDatosTrain, Datos * pDatosTest, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int funcionError){
	int countTrain = 0;

	// Inicialización de pesos
	pesosAleatorios();

	double minTrainError = 0;
	double minValidationError = 0;

	int numSinMejorar = 0;
	int numSinMejorarValidacion = 0;

	double testError = 0;
	double validationError = 0;

	nNumPatronesTrain = pDatosTrain->nNumPatrones;

	int *permutacion = NULL;

	Datos * pDatosValidation = new Datos;
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

		// Reservamos memoria para los datos de entrada y salida tanto de validacion como para la copia train
		pDatosValidation->entradas = new double*[pDatosValidation->nNumPatrones];
		pDatosValidation->salidas = new double*[pDatosValidation->nNumPatrones];

		pDatosTrain_copia->entradas = new double*[pDatosValidation->nNumPatrones];
		pDatosTrain_copia->salidas = new double*[pDatosValidation->nNumPatrones];

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
		bool isValidation = false;// Bandera para saber si pertenece a validacion

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
				for (int j = 0; j < pDatosValidation->nNumEntradas; j++)
				{
					pDatosValidation->entradas[validation_index][j] = pDatosTrain->entradas[i][j];
				}
				for (int j = 0; j < pDatosValidation->nNumSalidas; j++)
				{
					pDatosValidation->salidas[validation_index][j] = pDatosTrain->entradas[i][j];
				}
				validation_index++;
			}
			else{
				for (int j = 0; j < pDatosTrain_copia->nNumEntradas; j++)
				{
					pDatosTrain_copia->entradas[validation_index][j] = pDatosTrain->entradas[i][j];
				}
				for (int j = 0; j < pDatosTrain_copia->nNumSalidas; j++)
				{
					pDatosTrain_copia->salidas[validation_index][j] = pDatosTrain->entradas[i][j];
				}
			}
		}
		validationError = test(pDatosValidation, funcionError);
	}else{ // Si no hay validacion
		pDatosTrain_copia = pDatosTrain;
	}
	
	ofstream file("errores.txt");// Fichero para guardar los errores cometidos en cada iteracion
	ofstream fileCCR("ccr.txt");// Fichero para guardar el CCR conseguido en cada iteración
	// Aprendizaje del algoritmo
	do {

		entrenar(pDatosTrain,funcionError);

		double trainError = test(pDatosTrain,funcionError);
		// Gets the CCR with the train dataset per iteration
		*ccrTrain = testClassification(pDatosTrain);
		// Gets the CCR with the test dataset per iteration
		*ccrTest = testClassification(pDatosTest);
		if(countTrain==0 || trainError < minTrainError || fabs(trainError - minTrainError) > 0.00001){
			minTrainError = trainError;
			copiarPesos();
			numSinMejorar = 0;
		}
		else
			numSinMejorar++;

		if(numSinMejorar==50){
			cout << "Salida porque no mejora el entrenamiento!!"<< endl;
			restaurarPesos();
			countTrain = maxiter;
		}

		testError = test(pDatosTest,funcionError);
		countTrain++;

		// Comprobar condiciones de parada de validación para ver si terminamos la iteracion
		if (dValidacion>0 && dValidacion<1)
		{
			validationError = test(pDatosValidation, funcionError);
			if(countTrain==0 || validationError < minValidationError || fabs(validationError - minValidationError) > 0.00001){
				minValidationError = validationError;
				copiarPesos();
				numSinMejorarValidacion = 0;
			}else{
				numSinMejorarValidacion++;
			}
			if(numSinMejorarValidacion == 50){
				restaurarPesos();
				countTrain = maxiter;
			}
			cout << "Iteración " << countTrain << "\t Error de entrenamiento: " << trainError <<"\t Error de test: "<<testError<< "\t Error de validación: " << validationError << endl;
			file << countTrain<<","<< trainError<<","<<testError<<endl;
			fileCCR << countTrain <<","<<*ccrTrain<<","<<*ccrTest<<endl;
		}else{
			cout << "Iteración " << countTrain << "\t Error de entrenamiento: " << trainError << "\t Error de test: "<<testError<<endl;
			file << countTrain<<","<< trainError<<","<<testError<<endl;
			fileCCR << countTrain <<","<<*ccrTrain<<","<<*ccrTest<<endl;
		}
	} while ( countTrain<maxiter );

	restaurarPesos();

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
			cout << pDatosTest->salidas[i][j] << " -- " << prediccion[j] << " \\\\ " ;
		cout << endl;
		delete[] prediccion;

	}

	*errorTest=test(pDatosTest,funcionError);;
	*errorTrain=minTrainError;
	// Calculate the global CCR to return it to the main
	*ccrTest = testClassification(pDatosTest);
	*ccrTrain = testClassification(pDatosTrain);
	file.close();
	fileCCR.close();
}