package imglabeling;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.image.EncodeJpeg;
import org.tensorflow.op.io.ReadFile;
import org.tensorflow.op.io.WriteFile;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class ImgLblTensorFlow {

    // get path to model folder
    private static final String modelPath = "src/main/resources/models";

    public static void main(String[] params) {

        if (params.length != 2) {
            throw new IllegalArgumentException("Do not forget to pass the 2 params: input img and the output paths.");
        }
        //my output image
        String outputImagePath = params[1];
        // load saved model
        SavedModelBundle model = SavedModelBundle.load(modelPath, "serve");
        // create a map of the COCO 2017 labels (Common Objects in Context)
        TreeMap<Float, String> cocoTreeMap = new TreeMap<>();
        float cocoCount = 0;
        for (String cocoLabel : Labels.labels) {
            cocoTreeMap.put(cocoCount, cocoLabel);
            cocoCount++;
        }
        try (Graph g = new Graph(); Session s = new Session(g)) { // create graph, spin the session againt that graph
            Ops tensor = Ops.create(g); // create the working tensor
            Constant<TString> fileName = tensor.constant(params[0]); //my test image
            ReadFile readFile = tensor.io.readFile(fileName);
            Session.Runner runner = s.runner();
            DecodeJpeg.Options options = DecodeJpeg.channels(3L);
            DecodeJpeg decodeImage = tensor.image.decodeJpeg(readFile.contents(), options);

            Shape imageShape = runner.fetch(decodeImage).run().get(0).shape();  //fetch image from file

            //reshape the tensor to 4D for input to model
            Reshape<TUint8> reshape = tensor.reshape(decodeImage,
                    tensor.array(1,
                            imageShape.asArray()[0],
                            imageShape.asArray()[1],
                            imageShape.asArray()[2]
                    )
            );

            try (TUint8 reshapeTensor = (TUint8) s.runner().fetch(reshape).run().get(0)) {
                Map<String, Tensor> feedDict = new HashMap<>();
                // The given SavedModel SignatureDef input   https://www.tensorflow.org/tfx/serving/signature_defs
                feedDict.put("input_tensor", reshapeTensor);
                // The given SavedModel MetaGraphDef key https://www.tensorflow.org/api_docs/python/tf/compat/v1/MetaGraphDef
                Map<String, Tensor> outputTensorMap = model.function("serving_default").call(feedDict);
                // detection_classes, detectionBoxes etc. are model output names
                try (
                        TFloat32 detectionBoxes = (TFloat32) outputTensorMap.get("detection_boxes");
                        TFloat32 numDetections = (TFloat32) outputTensorMap.get("num_detections");
                        TFloat32 detectionScores = (TFloat32) outputTensorMap.get("detection_scores")
                ) {
                    int numDetects = (int) numDetections.getFloat(0);
                    if (numDetects > 0) {
                        ArrayList<FloatNdArray> boxArray = new ArrayList<>();
                        for (int n = 0; n < numDetects; n++) {
                            // put probability and position in outputMap
                            float detectionScore = detectionScores.getFloat(0, n);
                            //only include those classes with detection score greater than 0.3f
                            if (detectionScore > 0.3f) {
                                boxArray.add(detectionBoxes.get(0, n));
                            }
                        }
                        // (2D) RGBA colors to for the boxes
                        Operand<TFloat32> colors = tensor.constant(new float[][]{
                                {0.9f, 0.3f, 0.3f, 0.0f},
                                {0.3f, 0.3f, 0.9f, 0.0f},
                                {0.3f, 0.9f, 0.3f, 0.0f}
                        });
                        Shape boxesShape = Shape.of(1, boxArray.size(), 4);
                        int boxCount = 0;
                        // (3D) with shape `[batch, num_bounding_boxes, 4]` containing bounding boxes
                        try (TFloat32 boxes = TFloat32.tensorOf(boxesShape)) {
                            //batch size of 1
                            boxes.setFloat(1, 0, 0, 0);
                            for (FloatNdArray floatNdArray : boxArray) {
                                boxes.set(floatNdArray, 0, boxCount);
                                boxCount++;
                            }
                            // Placeholders for boxes and path to output mage
                            Placeholder<TFloat32> boxesPlaceHolder = tensor.placeholder(TFloat32.class, Placeholder.shape(boxesShape));
                            Placeholder<TString> outImagePathPlaceholder = tensor.placeholder(TString.class);
                            // Create JPEG from the Tensor with quality of 100%
                            EncodeJpeg.Options jpgOptions = EncodeJpeg.quality(100L);

                            WriteFile writeFile = tensor.io.writeFile(outImagePathPlaceholder,
                                    tensor.image.encodeJpeg( // convert the 4D input image to normalised 0.0f - 1.0f
                                            tensor.dtypes.cast(tensor.reshape(
                                                    tensor.math.mul(
                                                            tensor.image.drawBoundingBoxes(tensor.math.div( // Draw bounding boxes using boxes tensor and list of colors
                                                                    tensor.dtypes.cast(tensor.constant(reshapeTensor),
                                                                            TFloat32.class),
                                                                    tensor.constant(255.0f)
                                                                    ),
                                                                    boxesPlaceHolder, colors),
                                                            tensor.constant(255.0f)  // multiply by 255 then reshape and recast to TUint8 3D tensor
                                                    ),
                                                    tensor.array(
                                                            imageShape.asArray()[0],
                                                            imageShape.asArray()[1],
                                                            imageShape.asArray()[2]
                                                    )
                                            ), TUint8.class),
                                            jpgOptions));
                            // output the JPEG to file
                            s.runner().feed(outImagePathPlaceholder, TString.scalarOf(outputImagePath))
                                    .feed(boxesPlaceHolder, boxes)
                                    .addTarget(writeFile).run();
                        }
                    }
                }
            }
        }
    }

}
