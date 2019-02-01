package se.callistaenterprise.deeplearning4jdemo.app.service;

import com.google.common.primitives.Floats;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

@Service
public class DigitRecognizer {

    private static final String kerasModelPath = "/Users/davidstrom/Documents/ML/Python/mnist_digit_model.h5";
    private MultiLayerNetwork neuralNetwork;
    private final NativeImageLoader imageLoader = new NativeImageLoader(28, 28, 1);
    private final ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
    private List<Character> digits;

    @PostConstruct
    private void init() throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        neuralNetwork = KerasModelImport.importKerasSequentialModelAndWeights(kerasModelPath);
        digits = Arrays.asList('0', '1', '2', '3', '4', '5', '6', '7', '8', '9');
    }

    public char translateImageToDigit(InputStream image) throws IOException {
        final INDArray matrix = imageLoader.asMatrix(image);
        preProcessor.transform(matrix);
        final int[] prediction = neuralNetwork.predict(matrix);
        return digits.get(prediction[0]);
    }

    public HashMap<Character, Float> getDigitProbabilities(InputStream image) throws IOException {
        final INDArray matrix = imageLoader.asMatrix(image);
        preProcessor.transform(matrix);
        final INDArray output = neuralNetwork.output(matrix);
        HashMap<Character, Float> probabilities = new HashMap<>();
        final Iterator<Float> iterator = Floats.asList(output.getRow(0).data().dup().asFloat()).iterator();
        digits.stream().forEach(character -> probabilities.put(character, iterator.next()*100));
        return probabilities;
    }
}
