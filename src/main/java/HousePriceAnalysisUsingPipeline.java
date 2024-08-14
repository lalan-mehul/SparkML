import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

public class HousePriceAnalysisUsingPipeline {

    public static void main(String[] args) {
        long startTime = System.nanoTime();
        SparkSession sparkSession = SparkSession.builder().appName("HousePriceAnalysis").master("local[*]").getOrCreate();
        Dataset<Row> csvData
                = sparkSession.read().option("header", true)
                .option("inferSchema",true)
                .csv("src/main/resources/kc_house_data.csv");

        csvData = csvData.withColumn("sqft_above_percent", col("sqft_above").divide(col("sqft_living")))
                .withColumnRenamed("price","label");
        csvData = csvData.drop("id","date","view","sqft_lot","yr_renovated","lat","long","sqft_basement","sqft_lot15","yr_built","sqft_living15");

        Dataset<Row>[] datasets = csvData.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingAndTestDataSet = datasets[0];
        Dataset<Row> holdOutDataSet = datasets[1];

        StringIndexer conditionIndexer = new StringIndexer();
        conditionIndexer.setInputCol("condition");
        conditionIndexer.setOutputCol("conditionIndex");

        StringIndexer gradeIndexer = new StringIndexer();
        gradeIndexer.setInputCol("grade");
        gradeIndexer.setOutputCol("gradeIndex");

        StringIndexer zipCodeIndexer = new StringIndexer().setInputCol("zipcode").setOutputCol("zipcodeIndex");

        OneHotEncoder encoder = new OneHotEncoder().setInputCols(new String[]{"zipcodeIndex","conditionIndex","gradeIndex"})
                .setOutputCols(new String[]{"zipcodeVector","conditionVector","gradeVector"});


        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"bedrooms","bathrooms","sqft_living","sqft_above_percent","floors","waterfront","conditionVector","zipcodeVector","gradeVector"})
                .setOutputCol("features");

        LinearRegression linearRegression = new LinearRegression();
        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();

        ParamMap[] paramMaps = paramGridBuilder.addGrid(linearRegression.regParam(), new double[] {0.01, 0.1, 0.5})
                .addGrid(linearRegression.elasticNetParam(), new double[]{0,0.5,1})
                .build();

        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(linearRegression)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMaps).setTrainRatio(0.8);


        Pipeline pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[]{conditionIndexer, gradeIndexer, zipCodeIndexer, encoder, vectorAssembler, trainValidationSplit});
        PipelineModel pipelineModel = pipeline.fit(trainingAndTestDataSet);
        TrainValidationSplitModel model = (TrainValidationSplitModel) pipelineModel.stages()[5];
        LinearRegressionModel linearRegressionModel = (LinearRegressionModel) model.bestModel();

        Dataset<Row> holdOutResults = pipelineModel.transform(holdOutDataSet);
        holdOutResults.show();
        holdOutResults = holdOutResults.drop("prediction");
        //linearRegressionModel.transform(holdOutDataSet).show();

        System.out.println("The training data r2 value is: " + linearRegressionModel.summary().r2());
        System.out.println("The training data RMSE value is: " + linearRegressionModel.summary().rootMeanSquaredError());

        System.out.println("The holdout data r2 value is: " + linearRegressionModel.evaluate(holdOutResults).r2());
        System.out.println("The holdout data RMSE value is: " + linearRegressionModel.evaluate(holdOutResults).rootMeanSquaredError());

        System.out.println("Co-efficients:" + linearRegressionModel.coefficients());
        System.out.println("Intercept:" + linearRegressionModel.intercept());
        System.out.println("RegParam:" + linearRegressionModel.getRegParam());
        System.out.println("ElasticNetParam:" + linearRegressionModel.getElasticNetParam());


        System.out.println( "Time taken: " +  (System.nanoTime() -startTime)/1000000 ) ;


    }
}
