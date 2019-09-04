//============================================================================
// Introducción a los Modelos Computacionales
// Name        : practica1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // Para cojer la hora time()
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <string.h>
#include <math.h>
#include "imc/PerceptronMulticapa.h"

using namespace imc;
using namespace std;

int main(int argc, char **argv) {
	// Procesar los argumentos de la línea de comandos
    bool Tflag = 0, wflag = 0, pflag = 0;
    char *Tvalue = NULL, *wvalue = NULL;
    int c;

    //New arguments
    bool tflag = 0, iflag = 0, lflag = 0, hflag = 0, eflag = 0, mflag = 0, vflag = 0, dflag = 0, oflag = 0, fflag = 0, sflag = 0;// Flags
    char *tvalue = NULL; // File of the train data
    int ivalue = 1000; // Numer of iterations
    int lvalue = 1; // Number of hidden layers
    int hvalue = 5; // Number of neurons in each hidden layer
    double evalue = 0.5; // Value of eta param
    double mvalue = 0.9; // Value of mu param
    double vvalue = 0.0; // Value of the ratio
    double dvalue = 1.0; // Decrement factor

    opterr = 0;

    // a: opción que requiere un argumento
    // a:: el argumento requerido es opcional
    while ((c = getopt(argc, argv, "T:w:p:t:i:l:h:e:m:v:d::o::f::s")) != -1)
    {
        // Se han añadido los parámetros necesarios para usar el modo opcional de predicción (kaggle).
        // Añadir el resto de parámetros que sean necesarios para la parte básica de las prácticas.
        switch(c){
            case 'T':
                Tflag = true;
                Tvalue = optarg;
                break;
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'p':
                pflag = true;
                break;
            case 't':
                tflag = true;
                tvalue = optarg;
                break;
            case 'i':
                iflag = true;
                ivalue = atoi(optarg);
                break;
            case 'l':
                lflag = true;
                lvalue = atoi(optarg);
                break;
            case 'h':
                hflag = true;
                hvalue = atoi(optarg);
                break;
            case 'e':
                eflag = true;
                evalue = atof(optarg);
                break;
            case 'm':
                mflag = true;
                mvalue = atof(optarg);
                break;
            case 'v':
                vflag = true;
                vvalue = atof(optarg);
                break;
            case 'd':
                dflag = true;
                dvalue = atof(optarg);
                break;
            case 'o':
                oflag = true;
                break;
            case 'f':
                fflag = true;
                break;
            case 's':
                sflag = true;
                break;
            case '?':
                if (optopt == 'T' || optopt == 'w' || optopt == 'p' || optopt == 't' || optopt == 'i' || optopt == 'l' || optopt == 'h' || optopt == 'e' || optopt == 'm' || optopt == 'v' || optopt == 'd')
                    fprintf (stderr, "La opción -%c requiere un argumento.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Opción desconocida `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Caracter de opción desconocido `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }

 

    ////////////////////////////////////////
    // MODO DE ENTRENAMIENTO Y EVALUACIÓN //
    ///////////////////////////////////////
    // Control de errores
    // Si no se especifica el fichero train
    if(tflag == false){
        cout<<"Error! Se debe de especificar el fichero de entrenamiento"<<endl;
        return EXIT_FAILURE;
    }
    // Si no se especifica el fichero test
    if(Tflag == false){
        cout<<"Aviso! Se cogera el fichero de train para usarlo en el test"<<endl;
        Tvalue = tvalue;
    }
	// Objeto perceptrón multicapa
	PerceptronMulticapa mlp;

	// Lectura de datos de entrenamiento y test: llamar a mlp.leerDatos(...)
    Datos *pDatosTrain = mlp.leerDatos(tvalue);
    Datos *pDatosTest = mlp.leerDatos(Tvalue);

    // Parámetros del mlp. Por ejemplo, mlp.dEta = valorQueSea;
    mlp.dMu = mvalue;
    mlp.bOnline = oflag;
    mlp.dValidacion = vvalue;
    mlp.dDecremento = dvalue;

    if(oflag == true){ // Comprobamos si es modo online u offline
        mlp.dEta = evalue;
    }else{
        mlp.dEta = evalue/pDatosTrain->nNumPatrones;
    }

	// Inicializar vector topología
	int *topologia = new int[lvalue+2];
	topologia[0] = pDatosTrain->nNumEntradas;
	for(int i=1; i<(lvalue+2-1); i++)
		topologia[i] = hvalue;
	topologia[lvalue+2-1] = pDatosTrain->nNumSalidas;

	// Inicializar red con vector de topología
	mlp.inicializar(lvalue+2,topologia, sflag);


    // Semilla de los números aleatorios
    int semillas[] = {10,20,30,40,50};
    double *errores = new double[5];
    double *erroresTrain = new double[5];
    double *ccrs = new double[5];
    double *ccrsTrain = new double[5];
    double mejorErrorTest = 1.0;
    for(int i=0; i<5; i++){
    	cout << "**********" << endl;
    	cout << "SEMILLA " << semillas[i] << endl;
    	cout << "**********" << endl;
		srand(semillas[i]);
		mlp.ejecutarAlgoritmo(pDatosTrain,pDatosTest,ivalue,&(erroresTrain[i]),&(errores[i]),&(ccrsTrain[i]),&(ccrs[i]),fflag);
		cout << "Finalizamos => CCR de test final: " << ccrs[i] << endl;
    }

    cout<<"Datos iniciales: "<<endl;
    cout<<"\tHidden layers: "<<lvalue<<endl;
    cout<<"\tNeurons: "<<hvalue<<endl;
    cout<<"\teta value: "<<evalue<<endl;
    cout<<"\tMomentum value: "<<mvalue<<endl;
    cout<<"\tValidation value: "<<vvalue<<endl;
    cout<<"\tDecrement value: "<<dvalue<<endl;

    if(oflag == true){
        cout<<"\tModo online elegido"<<endl;
    }else{
        cout<<"\tModo offline elegido"<<endl;
    }
    if(sflag == true){
        cout<<"\tSoftmax elegida"<<endl;
    }else{
        cout<<"\tSigmoide elegida"<<endl;
    }
    if(fflag == true){
        cout<<"\tEntropia cruzada elegida"<<endl;
    }else{
        cout<<"\tMSE elegida"<<endl;
    }

    double mediaError = 0, desviacionTipicaError = 0;
    double mediaErrorTrain = 0, desviacionTipicaErrorTrain = 0;
    double mediaCCR = 0, desviacionTipicaCCR = 0;
    double mediaCCRTrain = 0, desviacionTipicaCCRTrain = 0;

    // Calcular medias y desviaciones típicas de entrenamiento y test
    for (int i = 0; i < 5; ++i)
    {
        mediaError += errores[i];
        mediaErrorTrain += erroresTrain[i];
        mediaCCR += ccrs[i];
        mediaCCRTrain += ccrsTrain[i];
    }
    mediaError /= 5;
    mediaErrorTrain /= 5;
    mediaCCR /= 5;
    mediaCCRTrain /= 5;

    for (int i = 0; i < 5; ++i)
    {
        desviacionTipicaError += pow(errores[i] - mediaError, 2);
        desviacionTipicaErrorTrain += pow(erroresTrain[i] - mediaErrorTrain, 2);
        desviacionTipicaCCR += pow(ccrs[i] - mediaCCR, 2);
        desviacionTipicaCCRTrain += pow(ccrsTrain[i] - mediaCCRTrain, 2);
    }
    desviacionTipicaError = sqrt(desviacionTipicaError/4);
    desviacionTipicaErrorTrain = sqrt(desviacionTipicaErrorTrain/4);
    desviacionTipicaCCR = sqrt(desviacionTipicaCCR/4);
    desviacionTipicaCCRTrain = sqrt(desviacionTipicaCCRTrain/4);

    cout << "HEMOS TERMINADO TODAS LAS SEMILLAS" << endl;

	cout << "INFORME FINAL" << endl;
	cout << "*************" << endl;
    cout << "Error de entrenamiento (Media +- DT): " << mediaErrorTrain << " +- " << desviacionTipicaErrorTrain << endl;
    cout << "Error de test (Media +- DT): " << mediaError << " +- " << desviacionTipicaError << endl;
    cout << "CCR de entrenamiento (Media +- DT): " << mediaCCRTrain << " +- " << desviacionTipicaCCRTrain << endl;
    cout << "CCR de test (Media +- DT): " << mediaCCR << " +- " << desviacionTipicaCCR << endl;
	return EXIT_SUCCESS;
}


