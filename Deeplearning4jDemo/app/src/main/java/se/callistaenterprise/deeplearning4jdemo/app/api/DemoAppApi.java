package se.callistaenterprise.deeplearning4jdemo.app.api;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import se.callistaenterprise.deeplearning4jdemo.app.service.DigitRecognizer;
import se.callistaenterprise.deeplearning4jdemo.app.service.CharacterRecognizer;

import javax.validation.Valid;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@RestController
@Validated
@RequestMapping("/api")
@EnableAutoConfiguration
public class DemoAppApi {

    @Autowired
    private CharacterRecognizer characterRecognizer;

    @Autowired
    private DigitRecognizer digitRecognizer;

    /*** DEMO PART ONE ***/

    @PostMapping("/character")
    public ResponseEntity<Map<String, Character>> evaluateCharacterDrawing(@Valid @RequestParam("file") MultipartFile file) throws IOException {
        Map<String, Character> result = new HashMap<>();
        final char character = characterRecognizer.translateImageToChar(file.getInputStream());
        result.put("You wrote", character);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/probabilities")
    public ResponseEntity<Map<Character, Float>> getCharacterProbabilities(@Valid @RequestParam("file") MultipartFile file) throws IOException {
        Map<Character, Float> probabilities = characterRecognizer.getCharacterProbabilities(file.getInputStream());
        return ResponseEntity.ok(probabilities);
    }

    /*** DEMO PART TWO ***/

    @PostMapping("/digit")
    public ResponseEntity<Map<String, Character>> evaluateDigitDrawing(@Valid @RequestParam("file") MultipartFile file) throws IOException {
        Map<String, Character> result = new HashMap<>();
        final char digit = digitRecognizer.translateImageToDigit(file.getInputStream());
        result.put("You wrote", digit);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/digitprob")
    public ResponseEntity<HashMap<Character, Float>> getDigitProbabilities(@Valid @RequestParam("file") MultipartFile file) throws IOException {
        HashMap<Character, Float> probabilities = digitRecognizer.getDigitProbabilities(file.getInputStream());
        return ResponseEntity.ok(probabilities);
    }

}
