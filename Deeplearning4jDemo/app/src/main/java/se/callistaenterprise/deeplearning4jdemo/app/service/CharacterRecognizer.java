package se.callistaenterprise.deeplearning4jdemo.app.service;

import com.google.common.primitives.Floats;
import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

@Slf4j
@Service
public class CharacterRecognizer {

    private MultiLayerNetwork neuralNetwork;

    private List<Character> latinCharacters;
    private final NativeImageLoader imageLoader = new NativeImageLoader(28, 28, 1);
    private final ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);

    @PostConstruct
    private void init() throws IOException {
        // Load trained ANN from disk
        neuralNetwork = ModelSerializer.restoreMultiLayerNetwork(new File(
                "/Users/davidstrom/Git/Deeplearning4jDemo/training/src/main/resources/models/cnn/bestModel.bin"), true);

        // Set of labels for our output
        latinCharacters = Arrays.asList('A', 'B', 'C', 'D', 'E', 'F', 'G',
                'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z');
    }

    public char translateImageToChar(InputStream image) throws IOException {
        // Read incoming image as a matrix of values
        final INDArray matrix = imageLoader.asMatrix(image);
        // Adjust pixel values from 0-255 -> 0-1
        preProcessor.transform(matrix);
        // Make prediction (returns index of activated node (neuron))
        final int[] prediction = neuralNetwork.predict(matrix);
        // Return character at 'that' index
        return latinCharacters.get(prediction[0]);
    }

    public HashMap<Character, Float> getCharacterProbabilities(InputStream image) throws IOException {
        // Read incoming image as a matrix of values
        final INDArray matrix = imageLoader.asMatrix(image);
        // Adjust pixel values from 0-255 -> 0-1
        preProcessor.transform(matrix);
        // Get output from entire output layer
        final INDArray output = neuralNetwork.output(matrix);

        HashMap<Character, Float> probabilities = new HashMap<>();
        final Iterator<Float> iterator = Floats.asList(output.getRow(0).data().dup().asFloat()).iterator();
        latinCharacters.stream().forEach(character -> probabilities.put(character, iterator.next()*100));
        return probabilities;
    }
}
