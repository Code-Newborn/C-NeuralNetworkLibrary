#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#define LIN_ACTIVATION 0x0001 /* Linear activation, i.e. output = input */
#define SIG_ACTIVATION 0x0002 /* Sigmoidal activation, i.e. output = 1/(1+e^(-input)) */
#define STP_ACTIVATION 0x0003 /* Step (not available yet) */

#define BPROP_LEARNING 0x0011 /* Back Propagation */
#define HEBB_LEARNING  0x0012 /* Hebbian learning (not available yet) */

#define SCALE_FOR_NET   0x0101 /* Scale human data for use in the network */
#define SCALE_FOR_HUMAN 0x0102 /* Scale network data for presentation to human */

typedef struct dataMember {
    double* inputs;  /* The input data */
    double* targets; /* The target outputs */
    double* outputs; /* The actual outputs */
    double* errors;  /* The error in the output */
} dataMember;

typedef struct dataset {
    dataMember* members;     /* The members of the dataset */
    double*     maxScale;    /* What the max value of the ins/outs are */
    double*     minScale;    /* What the min value of the ins/outs are */
    double*     sumSqErrors; /* The sum squared errors of all members */
    char*       name;        /* The name of the data set */
    int         numMembers;  /* The number of members in the set */
    int         numInputs;   /* The number of inputs in the set */
    int         numOutputs;  /* The number of outputs in the set */
} dataset;

typedef struct neuron {
    double   output;       /* The output of the neuron */
    double   delta;        /* The delta of the neuron */
    double** inputs;       /* An array of pointers to inputs */
    double*  weights;      /* An array of weights for the inputs + bias */
    double*  deltaWeights; /* An array of weight changes for inputs + bias */
    int      numInputs;    /* The number of inputs to the neuron */
    int      type;         /* The activation type of the neuron */
} neuron;

typedef struct mlpNetwork {
    neuron** layers;     /* Layers of neurons */
    int*     numNeurons; /* Number of neurons in each layer */
    int      numLayers;  /* The number of layers in the network */
    double   learnRate;  /* The learning rate of the neurons */
    double   momentum;   /* The momentum of the neurons */
    int      learning;   /* The learning type of the network */
    int      epoch;      /* The current epoch */
    int      epochMax;   /* The maximum number of epochs */
} mlpNetwork;

typedef struct dataset    dataset;
typedef struct mlpNetwork mlpNetwork;

dataset* loadData( char* filename, char* name );
void     destroyDataset( dataset* ptrDataset );

void setLearnParameters( mlpNetwork* Net, int emax, double learnRate, double momentum );
void setWeights( mlpNetwork* net, double* weights );
void computeNetwork( mlpNetwork* net, dataMember* datum, int numIn, int numOut );
void runNetworkOnce( mlpNetwork* net, dataset* data, int print );
void trainNetworkOnce( mlpNetwork* net, dataset* data, int print );


void        destroyNet( mlpNetwork* net );
mlpNetwork* createNetwork( int numLayers, int* numPerLayer, int inputs, int learnMethod, int defaultActivation );

#endif /* NEURAL_NETWORK_H */
