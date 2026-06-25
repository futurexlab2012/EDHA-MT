package com.example.futurexlab.edhamt.threads;

import com.example.futurexlab.edhamt.snn.SNN;
import com.example.futurexlab.edhamt.snn.inputUtils.InputQueue;
import com.example.futurexlab.edhamt.snn.logger.BasicLogger;
import com.example.futurexlab.edhamt.snn.logger.Logger;
import com.example.futurexlab.edhamt.snn.monitor.recorder.SpikeRecorder;
import com.example.futurexlab.edhamt.snn.neuron.Neuron;
import com.example.futurexlab.edhamt.snn.neuron.components.soma.LifFix;
import com.example.futurexlab.edhamt.snn.neuron.components.soma.Soma;
import com.example.futurexlab.edhamt.snn.queue.ComboQueue;
import com.example.futurexlab.edhamt.snn.queue.MinHeapQueue;


public class NNLib {

    public static Logger logger = new BasicLogger();


    public static Neuron[] CreateinputNeurons(int num){
        // input layer
        Neuron[] inputNeurons = new Neuron[num];
        for (int i = 0; i < inputNeurons.length; i++) {
            inputNeurons[i] = new Neuron(null);
        }
        return inputNeurons;
    }


    public static Neuron[] CreatefeaturesNeurons(int num,float threshold, float vReset, float tv, float tg, float refractoryTime, float threDelta, float threTime){
        // features layer
        Neuron[] featuresNeurons = new Neuron[num];
        for (int i = 0; i < featuresNeurons.length; i++) {
            Soma soma = new LifFix(threshold, vReset, tv, tg, refractoryTime , threDelta, threTime);
            featuresNeurons[i] = new Neuron(soma, "neuron " + i);
        }
        return  featuresNeurons;
    }


    public static InputQueue GenInputQueue(Neuron[] inputNeurons ){
        //Gen inputQueue
        InputQueue inputQueue = new InputQueue(inputNeurons);  //input neuron queue
        return  inputQueue;
    }


    public static SpikeRecorder GenSpikeRecorder(){
        //Gen SpikeRecorder
        MinHeapQueue<Neuron> neuronQueue = new MinHeapQueue<>(); //neuron manager queue
        SpikeRecorder spikeRecorder = new SpikeRecorder(neuronQueue);
        return spikeRecorder;
    }


    public static SNN GenSNNWithSpikes(SpikeRecorder spikeRecorder,InputQueue inputQueue){
        //Gen SNN Machine With  SpikeRecorder and InputQueue
        ComboQueue<Neuron> totalQueue = new ComboQueue<>(spikeRecorder, inputQueue);//combine the two queue
        SNN snn = new SNN(totalQueue, logger);
        return snn;
    }

}

