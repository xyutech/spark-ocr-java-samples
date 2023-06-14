package com.johnsnowlabs.ocr.samples.java;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import com.johnsnowlabs.ocr.ImageType;
import com.johnsnowlabs.ocr.transformers.BinaryToImage;
import com.johnsnowlabs.ocr.transformers.recognizers.ImageToTextV2;

import sparkocr.transformers.detectors.ImageTextDetectorV2;

public class ImageToTextV2Sample {
	
	public static PipelineModel getHandwrittenPipelineModel(SparkSession session) {
	    PipelineModel pipelineModel = null;
	    try {
	    	BinaryToImage binaryToImage = new BinaryToImage();
	        binaryToImage.setInputCol("content");
	        binaryToImage.setOutputCol("image");
	        binaryToImage.setImageType(ImageType.TYPE_3BYTE_BGR());

	        ImageTextDetectorV2 text_detector = 
	        		(ImageTextDetectorV2) ImageTextDetectorV2
	        		.pretrained("image_text_detector_v2", "en", "clinical/ocr");
	        text_detector.setInputCol("image") ;
	        text_detector.setOutputCol("text_regions") ;
	        text_detector.setSizeThreshold(-1) ;
	        text_detector.setLinkThreshold(0.3);
	        text_detector.setWidth(500);
	        text_detector.setWithRefiner(true);

	        ImageToTextV2 imageToTexV2t = 
	        		ImageToTextV2
	        		.pretrained("ocr_base_handwritten_v2_opt", "en", "clinical/ocr");
	        imageToTexV2t.setInputCols(new String[]{"image", "text_regions"}) ;
	        imageToTexV2t.setOutputCol("text");
	        imageToTexV2t.setGroupImages(true);
	        imageToTexV2t.setRegionsColumn("text_regions");

	        Pipeline pipeline = new Pipeline();
	        pipeline.setStages(new PipelineStage[] {
	        		binaryToImage,
	        		text_detector,
	        		imageToTexV2t
	        		});

	        pipelineModel = pipeline.fit(session.emptyDataFrame());
	        return pipelineModel;

	    } catch (Exception e) {
	        e.printStackTrace();
	        System.out.println(e.getLocalizedMessage());
	    }
	    return pipelineModel;
	}

	public static void main(String[] args) {
		SparkSession session = SessionBuilder.getSparkSession();
		String imgPath = ImageToTextV2Sample
				.class
				.getClassLoader()
				.getResource("handwritten_example.jpg")
				.getPath();
		Dataset<Row> df = session.read().format("binaryFile").load(imgPath);
		getHandwrittenPipelineModel(session)
		.transform(df)
		.show();
	}

}
