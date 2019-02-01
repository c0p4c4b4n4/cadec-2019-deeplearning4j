package se.callistaenterprise.deeplearning4jdemo.training;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.ClassificationScoreCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.weights.ConvolutionalIterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.deeplearning4j.nn.api.OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
import static org.deeplearning4j.nn.weights.WeightInit.XAVIER;
import static org.nd4j.linalg.activations.Activation.*;
import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;

@Slf4j
public class LatinCharacterRecognitionNetwork {

    private static final double learningRate = 0.0018;
    private static final int epocs = 30;
    private static final int numOfClasses = 26; // A -> Z
    private static final int randomSeed = 132;
    private static final int batchSize = 32;
    private static final double regularization = 0.0005;
    private static final int height = 28, width = 28, channels = 1;
    private static final String[] allowedExtensions = {".png"};
    private static final String modelSaveDirectory = "/Users/davidstrom/Git/Deeplearning4jDemo/training/src/main/resources/models/cnn";

    public static void main(String[] args) throws IOException, InterruptedException {
        File trainingDirectory = new File("/Users/davidstrom/Documents/ML/images/Latin/train");

        // Prepare training data
        ImageRecordReader trainingRecordReader = recordReader(trainingDirectory);
        DataSetIterator iterator = new RecordReaderDataSetIterator.Builder(trainingRecordReader, batchSize)
                .classification(1, numOfClasses)
                .preProcessor(new ImagePreProcessingScaler(0, 1)) // Scale pixel values to between 0 and 1 instead of 0 and 255
                .build();

        // Prepare neural network configuration
        MultiLayerConfiguration configuration = multiLayerConfiguration();
        // Create neural network with prepared configuration
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        // Prepare visualization of training
        prepareUIServer(model);
        EarlyStoppingTrainer trainer = trainer(iterator, model);
        EarlyStoppingResult result = trainer.fit();
        //Print out the results:
        log.info("Termination reason: {}", result.getTerminationReason());
        log.info("Total epochs: {}", result.getTotalEpochs());
        log.info("Best epoch number: {}", result.getBestModelEpoch());
        log.info("Score at best epoch: {}", result.getBestModelScore());
    }


    private static MultiLayerConfiguration multiLayerConfiguration() {
        int index = 0;
        return new NeuralNetConfiguration.Builder()
                .seed(randomSeed)
                .optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(XAVIER)
                .miniBatch(true)
                .updater(new Nesterovs(learningRate, 0.8))
                .convolutionMode(ConvolutionMode.Same)
                .l2(regularization)
                .list()
                .layer(index++, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(100)
                        .activation(IDENTITY)
                        .build())
                .layer(index++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(index++, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(IDENTITY)
                        .build())
                .layer(index++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(index++, new DenseLayer.Builder()
                        .nOut(150)
                        .activation(RELU)
                        .build())
                .layer(index++, new OutputLayer.Builder().lossFunction(NEGATIVELOGLIKELIHOOD)
                        .nOut(numOfClasses)
                        .activation(SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .pretrain(false)
                .backprop(true)
                .build();
    }

    private static EarlyStoppingTrainer trainer(DataSetIterator trainingDataIterator, MultiLayerNetwork model) {
        EarlyStoppingConfiguration earlyStoppingConfiguration = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(epocs))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(2, TimeUnit.HOURS))
                .scoreCalculator(new ClassificationScoreCalculator(org.nd4j.evaluation.classification.Evaluation.Metric.ACCURACY, trainingDataIterator/*Evaluation.Metric.ACCURACY, trainingDataIterator*/))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver(modelSaveDirectory))
                .build();
        return new EarlyStoppingTrainer(earlyStoppingConfiguration, model, trainingDataIterator);
    }

    private static ImageRecordReader recordReader(File directory) throws IOException {
        Random random = new Random(randomSeed);
        new FileSplit(directory, allowedExtensions);
        FileSplit fileSplit = new FileSplit(directory, allowedExtensions, random);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // Looks at parent directory name to set label
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(fileSplit);
        return recordReader;
    }

    private static void prepareUIServer(MultiLayerNetwork model) {
        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);
        model.addListeners(new StatsListener(storage));
    }

}
