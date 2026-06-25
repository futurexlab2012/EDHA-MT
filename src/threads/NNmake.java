package com.example.futurexlab.edhamt.threads;

import com.example.futurexlab.edhamt.snn.SNN;
import com.example.futurexlab.edhamt.snn.inputUtils.InputQueue;
import com.example.futurexlab.edhamt.snn.monitor.recorder.SpikeRecorder;
import com.example.futurexlab.edhamt.snn.neuron.Neuron;
import com.example.futurexlab.edhamt.snn.neuron.components.synapse.ConstantSynapse;
import com.example.futurexlab.edhamt.snn.neuron.components.synapse.LTDConstantSTDP;
import com.example.futurexlab.edhamt.snn.monitor.dataChannel.FileChannel;
import com.example.futurexlab.edhamt.snn.inputUtils.FileSpikeReader;
import com.example.futurexlab.edhamt.snn.monitor.recorder.SimpleRecorder;
import com.example.futurexlab.edhamt.snn.inputUtils.SpikePack;
import com.example.futurexlab.edhamt.snn.neuron.components.soma.LifFix;

import java.io.FileNotFoundException;


public class NNmake {

    public  static class SNNMachineRun1 extends Thread{

        private static int inputNeuronsnum;
        private static int featuresNeuronsnum;
        private static float threshold;
        private static float vReset;
        private static float tv;
        private static float tg;
        private static float refractoryTime;
        private static float threDelta;
        private static float threTime;
        private static LTDConstantSTDP.SharedVariate geAmplifyFactor;

        private static int snnid;
        private static int step;
        private static int epoch;

        public static NNLib snnlib = new NNLib();

        public SNNMachineRun1(int snnid,int step,int epoch,int inputNeuronsnum,int featuresNeuronsnum,float threshold,float vReset,float tv,float tg,float refractoryTime,float threDelta,float threTime,LTDConstantSTDP.SharedVariate geAmplifyFactor){
            this.snnid = snnid;
            this.step = step;
            this.epoch = epoch;
            this.inputNeuronsnum = inputNeuronsnum;
            this.featuresNeuronsnum = featuresNeuronsnum;
            this.threshold = threshold;
            this.vReset = vReset;
            this.tv = tv;
            this.tg = tg;
            this.refractoryTime = refractoryTime;
            this.threDelta = threDelta;
            this.threTime = threTime;
            this.geAmplifyFactor = geAmplifyFactor;
            System.out.println("ID:"+snnid+","+"SNN Machine is Running");
        }

        public static LTDConstantSTDP[][] CreateConnection(Neuron[] inputNeurons,Neuron[] outputNeurons){
            //connect neurons
            //input->output layer
            LTDConstantSTDP[][] input2outputSynapses = new LTDConstantSTDP[outputNeurons.length][inputNeurons.length];

            for (int i = 0; i < inputNeurons.length; i++) {
                for (int j = 0; j < outputNeurons.length; j++) {
                    LTDConstantSTDP synapse = new LTDConstantSTDP(Math.random() * 0.1 + 0.45, -0.2,
                            1.3, 20, -0.3, 0.015, geAmplifyFactor);
                    input2outputSynapses[j][i] = synapse;
                    inputNeurons[i].connect(outputNeurons[j], synapse);
                }
            }

            //features layer inhibition
            for (Neuron outputNeuron : outputNeurons) {
                for (Neuron neuron : outputNeurons) {
                    if (outputNeuron != neuron) {
                        outputNeuron.connect(neuron, new ConstantSynapse(-300));
                    }
                }
            }

            return input2outputSynapses;
        }

        @Override
        public  void  run() {

            Neuron[] inputNeurons = snnlib.CreateinputNeurons(inputNeuronsnum);
            Neuron[] featuresNeurons =  snnlib.CreatefeaturesNeurons(featuresNeuronsnum,threshold,vReset,tv,tg,refractoryTime,threDelta,threTime);

            LTDConstantSTDP[][] input2outputSynapses = CreateConnection(inputNeurons, featuresNeurons);
            for (LTDConstantSTDP[] input2outputSynaps : input2outputSynapses) {
                for (LTDConstantSTDP input2outputSynap : input2outputSynaps) {
                    input2outputSynap.setPlastic(true);
                }
            }

            InputQueue inputQueue = snnlib.GenInputQueue(inputNeurons);
            SpikeRecorder spikeRecorder = snnlib.GenSpikeRecorder();
            SNN snn = snnlib.GenSNNWithSpikes(spikeRecorder,inputQueue);

            //training
            double currentTime = 1e-5;
            SpikePack[] tempSpikePack = new SpikePack[1000];
            int spikePackWritePtr = 0;
            int spikePackReadPtr = 0;
            boolean hasSpike = true;
            SpikePack pack;

            for(int epochid = 0; epochid<epoch;epochid++) {

                spikePackWritePtr = 0;
                spikePackReadPtr = 0;

                FileSpikeReader fileSpikeReader = null;

                try {
                    if (fileSpikeReader != null) {
                        fileSpikeReader.close();
                    }
                    fileSpikeReader = new FileSpikeReader("data/training_images.spike");
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                    System.exit(1);
                }

                for (int i = 0; i < step ; i++) {
                    int inputSpikeCount = 0;
                    double maxSpikeTime = 0;

                    //load input spikes
                    while (true) {
                        if (hasSpike) {
                            pack = fileSpikeReader.read();
                            tempSpikePack[spikePackWritePtr++] = pack;  //write cache
                        } else {
                            pack = tempSpikePack[spikePackReadPtr++];   //read cache
                        }
                        if (pack.index < 0) {          //read a frame
                            break;
                        } else {
                            inputSpikeCount = inputSpikeCount+1;
                            if (pack.spikeTime > maxSpikeTime) {
                                maxSpikeTime = pack.spikeTime;
                            }
                            inputQueue.addSpike(new SpikePack(pack.index, pack.spikeTime + currentTime));
                        }
                    }

                    // run
                    currentTime = snn.run(200) + 50;

                    //update
                    if (spikeRecorder.nSpike() < 1) {
                        hasSpike = false;
                        spikePackReadPtr = 0;   //reset read point
                        geAmplifyFactor.value += 0.5;   //gain factor if no spikes
                        i--;  //retrain the sample
                    } else {
                        hasSpike = true;
                        spikePackWritePtr = 0;  //reset write point
                        //geAmplifyFactor.value = 0.5 + (0.002 * i);
                        geAmplifyFactor.value = 0.5;
                        System.out.println("Snnid:" + snnid + "," + "Step:" + i + "," + "NSpike=" + spikeRecorder.nSpike());
                    }
                    spikeRecorder.clear();  //clear spikes
                }
            }

            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            FileChannel fileChannel = new FileChannel();
            SimpleRecorder simpleRecorder = new SimpleRecorder(1024);
            simpleRecorder.clear();
            for (int j = 0; j < featuresNeurons.length; j++) {
                for (int k = 0; k < input2outputSynapses[j].length; k++) {
                    simpleRecorder.writeFloat((float) input2outputSynapses[j][k].getWeight());
                }
            }
            fileChannel.send(simpleRecorder.getData(), "data/out/" +"Snnid_"+snnid+"-Weight_"+step+".weight");  // save the final weight

            SimpleRecorder extraThreshold = new SimpleRecorder(featuresNeurons.length);
            for (Neuron featuresNeuron : featuresNeurons) {
                extraThreshold.writeDouble(((LifFix) featuresNeuron.getSoma()).getExtraThreshold());
            }
            fileChannel.send(extraThreshold.getData(), "data/out/"+"Snnid_"+snnid+"-ExtraThreshold_"+step+".double"); // update the potential threshold
        }
    }

    public  static class SNNMachineRun2 extends Thread{

        private static int inputNeuronsnum;
        private static int featuresNeuronsnum;
        private static float threshold;
        private static float vReset;
        private static float tv;
        private static float tg;
        private static float refractoryTime;
        private static float threDelta;
        private static float threTime;
        private static LTDConstantSTDP.SharedVariate geAmplifyFactor;

        private static int snnid;
        private static int step;
        private static int epoch;

        public static NNLib snnlib = new NNLib();

        public SNNMachineRun2(int snnid,int step,int epoch,int inputNeuronsnum,int featuresNeuronsnum,float threshold,float vReset,float tv,float tg,float refractoryTime,float threDelta,float threTime,LTDConstantSTDP.SharedVariate geAmplifyFactor){
            this.snnid = snnid;
            this.step = step;
            this.epoch = epoch;
            this.inputNeuronsnum = inputNeuronsnum;
            this.featuresNeuronsnum = featuresNeuronsnum;
            this.threshold = threshold;
            this.vReset = vReset;
            this.tv = tv;
            this.tg = tg;
            this.refractoryTime = refractoryTime;
            this.threDelta = threDelta;
            this.threTime = threTime;
            this.geAmplifyFactor = geAmplifyFactor;
            System.out.println("ID:"+snnid+","+"SNN Machine is Running");
        }

        public static LTDConstantSTDP[][] CreateConnection(Neuron[] inputNeurons,Neuron[] outputNeurons){
            //connect neurons
            //input->output layer
            LTDConstantSTDP[][] input2outputSynapses = new LTDConstantSTDP[outputNeurons.length][inputNeurons.length];

            for (int i = 0; i < inputNeurons.length; i++) {
                for (int j = 0; j < outputNeurons.length; j++) {
                    LTDConstantSTDP synapse = new LTDConstantSTDP(Math.random() * 0.1 + 0.45, -0.2,
                            1.3, 20, -0.3, 0.015, geAmplifyFactor);
                    input2outputSynapses[j][i] = synapse;
                    inputNeurons[i].connect(outputNeurons[j], synapse);
                }
            }

            //features layer inhibition
            for (Neuron outputNeuron : outputNeurons) {
                for (Neuron neuron : outputNeurons) {
                    if (outputNeuron != neuron) {
                        outputNeuron.connect(neuron, new ConstantSynapse(-300));
                    }
                }
            }

            return input2outputSynapses;
        }

        @Override
        public  void  run() {

            Neuron[] inputNeurons = snnlib.CreateinputNeurons(inputNeuronsnum);
            Neuron[] featuresNeurons =  snnlib.CreatefeaturesNeurons(featuresNeuronsnum,threshold,vReset,tv,tg,refractoryTime,threDelta,threTime);

            LTDConstantSTDP[][] input2outputSynapses = CreateConnection(inputNeurons, featuresNeurons);
            for (LTDConstantSTDP[] input2outputSynaps : input2outputSynapses) {
                for (LTDConstantSTDP input2outputSynap : input2outputSynaps) {
                    input2outputSynap.setPlastic(true);
                }
            }

            InputQueue inputQueue = snnlib.GenInputQueue(inputNeurons);
            SpikeRecorder spikeRecorder = snnlib.GenSpikeRecorder();
            SNN snn = snnlib.GenSNNWithSpikes(spikeRecorder,inputQueue);

            //training
            double currentTime = 1e-5;
            SpikePack[] tempSpikePack = new SpikePack[1000];
            int spikePackWritePtr = 0;
            int spikePackReadPtr = 0;
            boolean hasSpike = true;
            SpikePack pack;

            for(int epochid = 0; epochid<epoch;epochid++) {

                spikePackWritePtr = 0;
                spikePackReadPtr = 0;

                FileSpikeReader fileSpikeReader = null;

                try {
                    if (fileSpikeReader != null) {
                        fileSpikeReader.close();
                    }
                    fileSpikeReader = new FileSpikeReader("data/training_images.spike");
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                    System.exit(1);
                }

                for (int i = 0; i < step ; i++) {
                    int inputSpikeCount = 0;
                    double maxSpikeTime = 0;

                    //load input spikes
                    while (true) {
                        if (hasSpike) {
                            pack = fileSpikeReader.read();
                            tempSpikePack[spikePackWritePtr++] = pack;  //write cache
                        } else {
                            pack = tempSpikePack[spikePackReadPtr++];   //read cache
                        }
                        if (pack.index < 0) {          //read a frame
                            break;
                        } else {
                            inputSpikeCount = inputSpikeCount+1;
                            if (pack.spikeTime > maxSpikeTime) {
                                maxSpikeTime = pack.spikeTime;
                            }
                            inputQueue.addSpike(new SpikePack(pack.index, pack.spikeTime + currentTime));
                        }
                    }

                    // run
                    currentTime = snn.run(200) + 50;

                    //update
                    if (spikeRecorder.nSpike() < 1) {
                        hasSpike = false;
                        spikePackReadPtr = 0;   //reset read point
                        geAmplifyFactor.value += 0.5;   //gain factor if no spikes
                        i--;  //retrain the sample
                    } else {
                        hasSpike = true;
                        spikePackWritePtr = 0;  //reset write point

                        geAmplifyFactor.value = 0.5;
                        System.out.println("Snnid:" + snnid + "," + "Step:" + i + "," + "NSpike=" + spikeRecorder.nSpike());
                    }
                    spikeRecorder.clear();  //clear spikes
                }
            }

            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            FileChannel fileChannel = new FileChannel();
            SimpleRecorder simpleRecorder = new SimpleRecorder(1024);
            simpleRecorder.clear();
            for (int j = 0; j < featuresNeurons.length; j++) {
                for (int k = 0; k < input2outputSynapses[j].length; k++) {
                    simpleRecorder.writeFloat((float) input2outputSynapses[j][k].getWeight());
                }
            }
            fileChannel.send(simpleRecorder.getData(), "data/out/" +"Snnid_"+snnid+"-Weight_"+step+".weight");  // save the final weight

            SimpleRecorder extraThreshold = new SimpleRecorder(featuresNeurons.length);
            for (Neuron featuresNeuron : featuresNeurons) {
                extraThreshold.writeDouble(((LifFix) featuresNeuron.getSoma()).getExtraThreshold());
            }
            fileChannel.send(extraThreshold.getData(), "data/out/"+"Snnid_"+snnid+"-ExtraThreshold_"+step+".double"); // update the potential threshold
        }
    }

    public  static class SNNMachineRun3 extends Thread{

        private static int inputNeuronsnum;
        private static int featuresNeuronsnum;
        private static float threshold;
        private static float vReset;
        private static float tv;
        private static float tg;
        private static float refractoryTime;
        private static float threDelta;
        private static float threTime;
        private static LTDConstantSTDP.SharedVariate geAmplifyFactor;

        private static int snnid;
        private static int step;
        private static int epoch;

        public static NNLib snnlib = new NNLib();

        public SNNMachineRun3(int snnid,int step,int epoch,int inputNeuronsnum,int featuresNeuronsnum,float threshold,float vReset,float tv,float tg,float refractoryTime,float threDelta,float threTime,LTDConstantSTDP.SharedVariate geAmplifyFactor){
            this.snnid = snnid;
            this.step = step;
            this.epoch = epoch;
            this.inputNeuronsnum = inputNeuronsnum;
            this.featuresNeuronsnum = featuresNeuronsnum;
            this.threshold = threshold;
            this.vReset = vReset;
            this.tv = tv;
            this.tg = tg;
            this.refractoryTime = refractoryTime;
            this.threDelta = threDelta;
            this.threTime = threTime;
            this.geAmplifyFactor = geAmplifyFactor;
            System.out.println("ID:"+snnid+","+"SNN Machine is Running");
        }

        public static LTDConstantSTDP[][] CreateConnection(Neuron[] inputNeurons,Neuron[] outputNeurons){
            //connect neurons
            //input->output layer
            LTDConstantSTDP[][] input2outputSynapses = new LTDConstantSTDP[outputNeurons.length][inputNeurons.length];

            for (int i = 0; i < inputNeurons.length; i++) {
                for (int j = 0; j < outputNeurons.length; j++) {
                    LTDConstantSTDP synapse = new LTDConstantSTDP(Math.random() * 0.1 + 0.45, -0.2,
                            1.3, 20, -0.3, 0.015, geAmplifyFactor);
                    input2outputSynapses[j][i] = synapse;
                    inputNeurons[i].connect(outputNeurons[j], synapse);
                }
            }

            //features layer inhibition
            for (Neuron outputNeuron : outputNeurons) {
                for (Neuron neuron : outputNeurons) {
                    if (outputNeuron != neuron) {
                        outputNeuron.connect(neuron, new ConstantSynapse(-300));
                    }
                }
            }

            return input2outputSynapses;
        }

        @Override
        public  void  run() {

            Neuron[] inputNeurons = snnlib.CreateinputNeurons(inputNeuronsnum);
            Neuron[] featuresNeurons =  snnlib.CreatefeaturesNeurons(featuresNeuronsnum,threshold,vReset,tv,tg,refractoryTime,threDelta,threTime);

            LTDConstantSTDP[][] input2outputSynapses = CreateConnection(inputNeurons, featuresNeurons);
            for (LTDConstantSTDP[] input2outputSynaps : input2outputSynapses) {
                for (LTDConstantSTDP input2outputSynap : input2outputSynaps) {
                    input2outputSynap.setPlastic(true);
                }
            }

            InputQueue inputQueue = snnlib.GenInputQueue(inputNeurons);
            SpikeRecorder spikeRecorder = snnlib.GenSpikeRecorder();
            SNN snn = snnlib.GenSNNWithSpikes(spikeRecorder,inputQueue);

            //training
            double currentTime = 1e-5;
            SpikePack[] tempSpikePack = new SpikePack[1000];
            int spikePackWritePtr = 0;
            int spikePackReadPtr = 0;
            boolean hasSpike = true;
            SpikePack pack;

            for(int epochid = 0; epochid<epoch;epochid++) {

                spikePackWritePtr = 0;
                spikePackReadPtr = 0;

                FileSpikeReader fileSpikeReader = null;

                try {
                    if (fileSpikeReader != null) {
                        fileSpikeReader.close();
                    }
                    fileSpikeReader = new FileSpikeReader("data/training_images.spike");
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                    System.exit(1);
                }

                for (int i = 0; i < step ; i++) {
                    int inputSpikeCount = 0;
                    double maxSpikeTime = 0;

                    //load input spikes
                    while (true) {
                        if (hasSpike) {
                            pack = fileSpikeReader.read();
                            tempSpikePack[spikePackWritePtr++] = pack;  //write cache
                        } else {
                            pack = tempSpikePack[spikePackReadPtr++];   //read cache
                        }
                        if (pack.index < 0) {          //read a frame
                            break;
                        } else {
                            inputSpikeCount = inputSpikeCount+1;
                            if (pack.spikeTime > maxSpikeTime) {
                                maxSpikeTime = pack.spikeTime;
                            }
                            inputQueue.addSpike(new SpikePack(pack.index, pack.spikeTime + currentTime));
                        }
                    }

                    // run
                    currentTime = snn.run(200) + 50;

                    //update
                    if (spikeRecorder.nSpike() < 1) {
                        hasSpike = false;
                        spikePackReadPtr = 0;   //reset read point
                        geAmplifyFactor.value += 0.5;   //gain factor if no spikes
                        i--;  //retrain the sample
                    } else {
                        hasSpike = true;
                        spikePackWritePtr = 0;  //reset write point

                        geAmplifyFactor.value = 0.5;
                        System.out.println("Snnid:" + snnid + "," + "Step:" + i + "," + "NSpike=" + spikeRecorder.nSpike());
                    }
                    spikeRecorder.clear();  //clear spikes
                }
            }

            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            FileChannel fileChannel = new FileChannel();
            SimpleRecorder simpleRecorder = new SimpleRecorder(1024);
            simpleRecorder.clear();
            for (int j = 0; j < featuresNeurons.length; j++) {
                for (int k = 0; k < input2outputSynapses[j].length; k++) {
                    simpleRecorder.writeFloat((float) input2outputSynapses[j][k].getWeight());
                }
            }
            fileChannel.send(simpleRecorder.getData(), "data/out/" +"Snnid_"+snnid+"-Weight_"+step+".weight");  // save the final weight

            SimpleRecorder extraThreshold = new SimpleRecorder(featuresNeurons.length);
            for (Neuron featuresNeuron : featuresNeurons) {
                extraThreshold.writeDouble(((LifFix) featuresNeuron.getSoma()).getExtraThreshold());
            }
            fileChannel.send(extraThreshold.getData(), "data/out/"+"Snnid_"+snnid+"-ExtraThreshold_"+step+".double"); // update the potential threshold
        }
    }

    public  static class SNNMachineRun4 extends Thread{

        private static int inputNeuronsnum;
        private static int featuresNeuronsnum;
        private static float threshold;
        private static float vReset;
        private static float tv;
        private static float tg;
        private static float refractoryTime;
        private static float threDelta;
        private static float threTime;
        private static LTDConstantSTDP.SharedVariate geAmplifyFactor;

        private static int snnid;
        private static int step;
        private static int epoch;

        public static NNLib snnlib = new NNLib();

        public SNNMachineRun4(int snnid,int step,int epoch,int inputNeuronsnum,int featuresNeuronsnum,float threshold,float vReset,float tv,float tg,float refractoryTime,float threDelta,float threTime,LTDConstantSTDP.SharedVariate geAmplifyFactor){
            this.snnid = snnid;
            this.step = step;
            this.epoch = epoch;
            this.inputNeuronsnum = inputNeuronsnum;
            this.featuresNeuronsnum = featuresNeuronsnum;
            this.threshold = threshold;
            this.vReset = vReset;
            this.tv = tv;
            this.tg = tg;
            this.refractoryTime = refractoryTime;
            this.threDelta = threDelta;
            this.threTime = threTime;
            this.geAmplifyFactor = geAmplifyFactor;
            System.out.println("ID:"+snnid+","+"SNN Machine is Running");
        }

        public static LTDConstantSTDP[][] CreateConnection(Neuron[] inputNeurons,Neuron[] outputNeurons){
            //connect neurons
            //input->output layer
            LTDConstantSTDP[][] input2outputSynapses = new LTDConstantSTDP[outputNeurons.length][inputNeurons.length];

            for (int i = 0; i < inputNeurons.length; i++) {
                for (int j = 0; j < outputNeurons.length; j++) {
                    LTDConstantSTDP synapse = new LTDConstantSTDP(Math.random() * 0.1 + 0.45, -0.2,
                            1.3, 20, -0.3, 0.015, geAmplifyFactor);
                    input2outputSynapses[j][i] = synapse;
                    inputNeurons[i].connect(outputNeurons[j], synapse);
                }
            }

            //features layer inhibition
            for (Neuron outputNeuron : outputNeurons) {
                for (Neuron neuron : outputNeurons) {
                    if (outputNeuron != neuron) {
                        outputNeuron.connect(neuron, new ConstantSynapse(-300));
                    }
                }
            }

            return input2outputSynapses;
        }

        @Override
        public  void  run() {

            Neuron[] inputNeurons = snnlib.CreateinputNeurons(inputNeuronsnum);
            Neuron[] featuresNeurons =  snnlib.CreatefeaturesNeurons(featuresNeuronsnum,threshold,vReset,tv,tg,refractoryTime,threDelta,threTime);

            LTDConstantSTDP[][] input2outputSynapses = CreateConnection(inputNeurons, featuresNeurons);
            for (LTDConstantSTDP[] input2outputSynaps : input2outputSynapses) {
                for (LTDConstantSTDP input2outputSynap : input2outputSynaps) {
                    input2outputSynap.setPlastic(true);
                }
            }

            InputQueue inputQueue = snnlib.GenInputQueue(inputNeurons);
            SpikeRecorder spikeRecorder = snnlib.GenSpikeRecorder();
            SNN snn = snnlib.GenSNNWithSpikes(spikeRecorder,inputQueue);

            //training
            double currentTime = 1e-5;
            SpikePack[] tempSpikePack = new SpikePack[1000];
            int spikePackWritePtr = 0;
            int spikePackReadPtr = 0;
            boolean hasSpike = true;
            SpikePack pack;

            for(int epochid = 0; epochid<epoch;epochid++) {

                spikePackWritePtr = 0;
                spikePackReadPtr = 0;

                FileSpikeReader fileSpikeReader = null;

                try {
                    if (fileSpikeReader != null) {
                        fileSpikeReader.close();
                    }
                    fileSpikeReader = new FileSpikeReader("data/training_images.spike");
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                    System.exit(1);
                }

                for (int i = 0; i < step ; i++) {
                    int inputSpikeCount = 0;
                    double maxSpikeTime = 0;

                    //load input spikes
                    while (true) {
                        if (hasSpike) {
                            pack = fileSpikeReader.read();
                            tempSpikePack[spikePackWritePtr++] = pack;  //write cache
                        } else {
                            pack = tempSpikePack[spikePackReadPtr++];   //read cache
                        }
                        if (pack.index < 0) {          //read a frame
                            break;
                        } else {
                            inputSpikeCount = inputSpikeCount+1;
                            if (pack.spikeTime > maxSpikeTime) {
                                maxSpikeTime = pack.spikeTime;
                            }
                            inputQueue.addSpike(new SpikePack(pack.index, pack.spikeTime + currentTime));
                        }
                    }

                    // run
                    currentTime = snn.run(200) + 50;

                    //update
                    if (spikeRecorder.nSpike() < 1) {
                        hasSpike = false;
                        spikePackReadPtr = 0;   //reset read point
                        geAmplifyFactor.value += 0.5;   //gain factor if no spikes
                        i--;  //retrain the sample
                    } else {
                        hasSpike = true;
                        spikePackWritePtr = 0;  //reset write point

                        geAmplifyFactor.value = 0.5;
                        System.out.println("Snnid:" + snnid + "," + "Step:" + i + "," + "NSpike=" + spikeRecorder.nSpike());
                    }
                    spikeRecorder.clear();  //clear spikes
                }
            }

            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            FileChannel fileChannel = new FileChannel();
            SimpleRecorder simpleRecorder = new SimpleRecorder(1024);
            simpleRecorder.clear();
            for (int j = 0; j < featuresNeurons.length; j++) {
                for (int k = 0; k < input2outputSynapses[j].length; k++) {
                    simpleRecorder.writeFloat((float) input2outputSynapses[j][k].getWeight());
                }
            }
            fileChannel.send(simpleRecorder.getData(), "data/out/" +"Snnid_"+snnid+"-Weight_"+step+".weight");  // save the final weight

            SimpleRecorder extraThreshold = new SimpleRecorder(featuresNeurons.length);
            for (Neuron featuresNeuron : featuresNeurons) {
                extraThreshold.writeDouble(((LifFix) featuresNeuron.getSoma()).getExtraThreshold());
            }
            fileChannel.send(extraThreshold.getData(), "data/out/"+"Snnid_"+snnid+"-ExtraThreshold_"+step+".double"); // update the potential threshold
        }
    }

    public  static class SNNMachineRun5 extends Thread{

        private static int inputNeuronsnum;
        private static int featuresNeuronsnum;
        private static float threshold;
        private static float vReset;
        private static float tv;
        private static float tg;
        private static float refractoryTime;
        private static float threDelta;
        private static float threTime;
        private static LTDConstantSTDP.SharedVariate geAmplifyFactor;

        private static int snnid;
        private static int step;
        private static int epoch;

        public static NNLib snnlib = new NNLib();

        public SNNMachineRun5(int snnid,int step,int epoch,int inputNeuronsnum,int featuresNeuronsnum,float threshold,float vReset,float tv,float tg,float refractoryTime,float threDelta,float threTime,LTDConstantSTDP.SharedVariate geAmplifyFactor){
            this.snnid = snnid;
            this.step = step;
            this.epoch = epoch;
            this.inputNeuronsnum = inputNeuronsnum;
            this.featuresNeuronsnum = featuresNeuronsnum;
            this.threshold = threshold;
            this.vReset = vReset;
            this.tv = tv;
            this.tg = tg;
            this.refractoryTime = refractoryTime;
            this.threDelta = threDelta;
            this.threTime = threTime;
            this.geAmplifyFactor = geAmplifyFactor;
            System.out.println("ID:"+snnid+","+"SNN Machine is Running");
        }

        public static LTDConstantSTDP[][] CreateConnection(Neuron[] inputNeurons,Neuron[] outputNeurons){
            //connect neurons
            //input->output layer
            LTDConstantSTDP[][] input2outputSynapses = new LTDConstantSTDP[outputNeurons.length][inputNeurons.length];

            for (int i = 0; i < inputNeurons.length; i++) {
                for (int j = 0; j < outputNeurons.length; j++) {
                    LTDConstantSTDP synapse = new LTDConstantSTDP(Math.random() * 0.1 + 0.45, -0.2,
                            1.3, 20, -0.3, 0.015, geAmplifyFactor);
                    input2outputSynapses[j][i] = synapse;
                    inputNeurons[i].connect(outputNeurons[j], synapse);
                }
            }

            //features layer inhibition
            for (Neuron outputNeuron : outputNeurons) {
                for (Neuron neuron : outputNeurons) {
                    if (outputNeuron != neuron) {
                        outputNeuron.connect(neuron, new ConstantSynapse(-300));
                    }
                }
            }

            return input2outputSynapses;
        }

        @Override
        public  void  run() {

            Neuron[] inputNeurons = snnlib.CreateinputNeurons(inputNeuronsnum);
            Neuron[] featuresNeurons =  snnlib.CreatefeaturesNeurons(featuresNeuronsnum,threshold,vReset,tv,tg,refractoryTime,threDelta,threTime);

            LTDConstantSTDP[][] input2outputSynapses = CreateConnection(inputNeurons, featuresNeurons);
            for (LTDConstantSTDP[] input2outputSynaps : input2outputSynapses) {
                for (LTDConstantSTDP input2outputSynap : input2outputSynaps) {
                    input2outputSynap.setPlastic(true);//每个突触均有可塑性
                }
            }

            InputQueue inputQueue = snnlib.GenInputQueue(inputNeurons);
            SpikeRecorder spikeRecorder = snnlib.GenSpikeRecorder();
            SNN snn = snnlib.GenSNNWithSpikes(spikeRecorder,inputQueue);

            //training
            double currentTime = 1e-5;
            SpikePack[] tempSpikePack = new SpikePack[1000];
            int spikePackWritePtr = 0;
            int spikePackReadPtr = 0;
            boolean hasSpike = true;
            SpikePack pack;

            for(int epochid = 0; epochid<epoch;epochid++) {

                spikePackWritePtr = 0;
                spikePackReadPtr = 0;

                FileSpikeReader fileSpikeReader = null;

                try {
                    if (fileSpikeReader != null) {
                        fileSpikeReader.close();
                    }
                    fileSpikeReader = new FileSpikeReader("data/training_images.spike");
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                    System.exit(1);
                }

                for (int i = 0; i < step ; i++) {
                    int inputSpikeCount = 0;
                    double maxSpikeTime = 0;

                    //load input spikes
                    while (true) {
                        if (hasSpike) {
                            pack = fileSpikeReader.read();
                            tempSpikePack[spikePackWritePtr++] = pack;  //write cache
                        } else {
                            pack = tempSpikePack[spikePackReadPtr++];   //read cache
                        }
                        if (pack.index < 0) {          //read a frame
                            break;
                        } else {
                            inputSpikeCount = inputSpikeCount+1;
                            if (pack.spikeTime > maxSpikeTime) {
                                maxSpikeTime = pack.spikeTime;
                            }
                            inputQueue.addSpike(new SpikePack(pack.index, pack.spikeTime + currentTime));
                        }
                    }

                    // run
                    currentTime = snn.run(200) + 50;

                    //update
                    if (spikeRecorder.nSpike() < 1) {
                        hasSpike = false;
                        spikePackReadPtr = 0;   //reset read point
                        geAmplifyFactor.value += 0.5;   //gain factor if no spikes
                        i--;  //retrain the sample
                    } else {
                        hasSpike = true;
                        spikePackWritePtr = 0;  //reset write point

                        geAmplifyFactor.value = 0.5;
                        System.out.println("Snnid:" + snnid + "," + "Step:" + i + "," + "NSpike=" + spikeRecorder.nSpike());
                    }
                    spikeRecorder.clear();  //clear spikes
                }
            }

            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            FileChannel fileChannel = new FileChannel();
            SimpleRecorder simpleRecorder = new SimpleRecorder(1024);
            simpleRecorder.clear();
            for (int j = 0; j < featuresNeurons.length; j++) {
                for (int k = 0; k < input2outputSynapses[j].length; k++) {
                    simpleRecorder.writeFloat((float) input2outputSynapses[j][k].getWeight());
                }
            }
            fileChannel.send(simpleRecorder.getData(), "data/out/" +"Snnid_"+snnid+"-Weight_"+step+".weight");  // save the final weight

            SimpleRecorder extraThreshold = new SimpleRecorder(featuresNeurons.length);
            for (Neuron featuresNeuron : featuresNeurons) {
                extraThreshold.writeDouble(((LifFix) featuresNeuron.getSoma()).getExtraThreshold());
            }
            fileChannel.send(extraThreshold.getData(), "data/out/"+"Snnid_"+snnid+"-ExtraThreshold_"+step+".double"); // update the potential threshold
        }
    }

    public  static class SNNMachineRun6 extends Thread{

        private static int inputNeuronsnum;
        private static int featuresNeuronsnum;
        private static float threshold;
        private static float vReset;
        private static float tv;
        private static float tg;
        private static float refractoryTime;
        private static float threDelta;
        private static float threTime;
        private static LTDConstantSTDP.SharedVariate geAmplifyFactor;

        private static int snnid;
        private static int step;
        private static int epoch;

        public static NNLib snnlib = new NNLib();

        public SNNMachineRun6(int snnid,int step,int epoch,int inputNeuronsnum,int featuresNeuronsnum,float threshold,float vReset,float tv,float tg,float refractoryTime,float threDelta,float threTime,LTDConstantSTDP.SharedVariate geAmplifyFactor){
            this.snnid = snnid;
            this.step = step;
            this.epoch = epoch;
            this.inputNeuronsnum = inputNeuronsnum;
            this.featuresNeuronsnum = featuresNeuronsnum;
            this.threshold = threshold;
            this.vReset = vReset;
            this.tv = tv;
            this.tg = tg;
            this.refractoryTime = refractoryTime;
            this.threDelta = threDelta;
            this.threTime = threTime;
            this.geAmplifyFactor = geAmplifyFactor;
            System.out.println("ID:"+snnid+","+"SNN Machine is Running");
        }

        public static LTDConstantSTDP[][] CreateConnection(Neuron[] inputNeurons,Neuron[] outputNeurons){
            //connect neurons
            //input->output layer
            LTDConstantSTDP[][] input2outputSynapses = new LTDConstantSTDP[outputNeurons.length][inputNeurons.length];

            for (int i = 0; i < inputNeurons.length; i++) {
                for (int j = 0; j < outputNeurons.length; j++) {
                    LTDConstantSTDP synapse = new LTDConstantSTDP(Math.random() * 0.1 + 0.45, -0.2,
                            1.3, 20, -0.3, 0.015, geAmplifyFactor);
                    input2outputSynapses[j][i] = synapse;
                    inputNeurons[i].connect(outputNeurons[j], synapse);
                }
            }

            //features layer inhibition
            for (Neuron outputNeuron : outputNeurons) {
                for (Neuron neuron : outputNeurons) {
                    if (outputNeuron != neuron) {
                        outputNeuron.connect(neuron, new ConstantSynapse(-300));
                    }
                }
            }

            return input2outputSynapses;
        }

        @Override
        public  void  run() {

            Neuron[] inputNeurons = snnlib.CreateinputNeurons(inputNeuronsnum);
            Neuron[] featuresNeurons =  snnlib.CreatefeaturesNeurons(featuresNeuronsnum,threshold,vReset,tv,tg,refractoryTime,threDelta,threTime);

            LTDConstantSTDP[][] input2outputSynapses = CreateConnection(inputNeurons, featuresNeurons);
            for (LTDConstantSTDP[] input2outputSynaps : input2outputSynapses) {
                for (LTDConstantSTDP input2outputSynap : input2outputSynaps) {
                    input2outputSynap.setPlastic(true);
                }
            }

            InputQueue inputQueue = snnlib.GenInputQueue(inputNeurons);
            SpikeRecorder spikeRecorder = snnlib.GenSpikeRecorder();
            SNN snn = snnlib.GenSNNWithSpikes(spikeRecorder,inputQueue);

            //training
            double currentTime = 1e-5;
            SpikePack[] tempSpikePack = new SpikePack[1000];
            int spikePackWritePtr = 0;
            int spikePackReadPtr = 0;
            boolean hasSpike = true;
            SpikePack pack;

            for(int epochid = 0; epochid<epoch;epochid++) {

                spikePackWritePtr = 0;
                spikePackReadPtr = 0;

                FileSpikeReader fileSpikeReader = null;

                try {
                    if (fileSpikeReader != null) {
                        fileSpikeReader.close();
                    }
                    fileSpikeReader = new FileSpikeReader("data/training_images.spike");
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                    System.exit(1);
                }

                for (int i = 0; i < step ; i++) {
                    int inputSpikeCount = 0;
                    double maxSpikeTime = 0;

                    //load input spikes
                    while (true) {
                        if (hasSpike) {
                            pack = fileSpikeReader.read();
                            tempSpikePack[spikePackWritePtr++] = pack;  //write cache
                        } else {
                            pack = tempSpikePack[spikePackReadPtr++];   //read cache
                        }
                        if (pack.index < 0) {          //read a frame
                            break;
                        } else {
                            inputSpikeCount = inputSpikeCount+1;
                            if (pack.spikeTime > maxSpikeTime) {
                                maxSpikeTime = pack.spikeTime;
                            }
                            inputQueue.addSpike(new SpikePack(pack.index, pack.spikeTime + currentTime));
                        }
                    }

                    // run
                    currentTime = snn.run(200) + 50;

                    //update
                    if (spikeRecorder.nSpike() < 1) {
                        hasSpike = false;
                        spikePackReadPtr = 0;   //reset read point
                        geAmplifyFactor.value += 0.5;   //gain factor if no spikes
                        i--;  //retrain the sample
                    } else {
                        hasSpike = true;
                        spikePackWritePtr = 0;  //reset write point

                        geAmplifyFactor.value = 0.5;
                        System.out.println("Snnid:" + snnid + "," + "Step:" + i + "," + "NSpike=" + spikeRecorder.nSpike());
                    }
                    spikeRecorder.clear();  //clear spikes
                }
            }

            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            FileChannel fileChannel = new FileChannel();
            SimpleRecorder simpleRecorder = new SimpleRecorder(1024);
            simpleRecorder.clear();
            for (int j = 0; j < featuresNeurons.length; j++) {
                for (int k = 0; k < input2outputSynapses[j].length; k++) {
                    simpleRecorder.writeFloat((float) input2outputSynapses[j][k].getWeight());
                }
            }
            fileChannel.send(simpleRecorder.getData(), "data/out/" +"Snnid_"+snnid+"-Weight_"+step+".weight");  // save the final weight

            SimpleRecorder extraThreshold = new SimpleRecorder(featuresNeurons.length);
            for (Neuron featuresNeuron : featuresNeurons) {
                extraThreshold.writeDouble(((LifFix) featuresNeuron.getSoma()).getExtraThreshold());
            }
            fileChannel.send(extraThreshold.getData(), "data/out/"+"Snnid_"+snnid+"-ExtraThreshold_"+step+".double"); // update the potential threshold
        }
    }

    public static void main(String[] args) throws InterruptedException {


        long startime = System.currentTimeMillis();

        SNNMachineRun1 snn1 = new SNNMachineRun1(1,60000,1,785,100,-10, -85, 50, 5, 20, 2.5f, 1e7f, new LTDConstantSTDP.SharedVariate(0.5));
        //snn1.run(); /


        SNNMachineRun2 snn2= new SNNMachineRun2(2,60000,1,785,100,-10, -85, 50, 5, 20, 2.5f, 1e7f, new LTDConstantSTDP.SharedVariate(0.5));
        SNNMachineRun3 snn3= new SNNMachineRun3(3,60000,1,785,100,-10, -85, 50, 5, 20, 2.5f, 1e7f, new LTDConstantSTDP.SharedVariate(0.5));
        SNNMachineRun4 snn4= new SNNMachineRun4(4,60000,1,785,100,-10, -85, 50, 5, 20, 2.5f, 1e7f, new LTDConstantSTDP.SharedVariate(0.5));
        SNNMachineRun5 snn5= new SNNMachineRun5(5,60000,1,785,100,-10, -85, 50, 5, 20, 2.5f, 1e7f, new LTDConstantSTDP.SharedVariate(0.5));
        SNNMachineRun6 snn6= new SNNMachineRun6(6,60000,1,785,100,-10, -85, 50, 5, 20, 2.5f, 1e7f, new LTDConstantSTDP.SharedVariate(0.5));
        snn1.start();
        snn2.start();
//        snn3.start();
//        snn4.start();
//        snn5.start();
//        snn6.start();
        snn1.join();
        snn2.join();
//        snn3.join();
//        snn4.join();
//        snn5.join();
//        snn6.join();
        snn1.interrupt();
        snn2.interrupt();
//        snn3.interrupt();
//        snn4.interrupt();
//        snn5.interrupt();
//        snn6.interrupt();


        long endtime = System.currentTimeMillis();
        System.out.println("Time cost:" + (endtime - startime)/1000 + 'S' );
    }
}
