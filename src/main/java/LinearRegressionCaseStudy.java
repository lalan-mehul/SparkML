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
import static org.apache.spark.sql.functions.*;

public class LinearRegressionCaseStudy {

    public static void main(String[] args) {
        long startTime = System.nanoTime();
        SparkSession sparkSession = SparkSession.builder().appName("LinearRegressionCaseStudy").master("local[*]").getOrCreate();
        Dataset<Row> csvData
                = sparkSession.read().option("header", true)
                .option("inferSchema",true)
                .csv("src/main/resources/vppChapterViews/part-r-*.csv");
        csvData = csvData.filter("is_cancelled=false");
        csvData = csvData.drop("observation_date","is_cancelled");

        String[] columnsThatNeedNullReplacement = new String[]{"firstSub","all_time_views","last_month_views","next_month_views"};
        for (String column : columnsThatNeedNullReplacement) {
            csvData = csvData.withColumn(column, when(col(column).isNull(), 0).otherwise(col(column)));
        }

        StringIndexer payment_method_type_indexer = new StringIndexer().setInputCol("payment_method_type").setOutputCol("payment_method_type_indexer");
        csvData = payment_method_type_indexer.fit(csvData).transform(csvData);
        StringIndexer country_indexer = new StringIndexer().setInputCol("country").setOutputCol("country_indexer");
        csvData = country_indexer.fit(csvData).transform(csvData);
        StringIndexer rebill_period_indexer = new StringIndexer().setInputCol("rebill_period_in_months").setOutputCol("rebill_period_in_months_indexer");
        csvData = rebill_period_indexer.fit(csvData).transform(csvData);

        OneHotEncoder encoder = new OneHotEncoder();
        encoder.setInputCols(new String[]{"payment_method_type_indexer","country_indexer","rebill_period_in_months_indexer"});
        encoder.setOutputCols(new String[]{"payment_method_type_encoder","country_encoder","rebill_period_in_months_encoder"});
        csvData = encoder.fit(csvData).transform(csvData);

        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[]{"firstSub","age","all_time_views","last_month_views","payment_method_type_encoder","country_encoder","rebill_period_in_months_encoder"}).setOutputCol("features");
        csvData = vectorAssembler.transform(csvData).select("next_month_views","features").withColumnRenamed("next_month_views","label");

        Dataset<Row>[] datasets = csvData.randomSplit(new double[]{0.9, 0.1});
        Dataset<Row> trainingAndTestDataSet = datasets[0];
        Dataset<Row> holdOutDataSet = datasets[1];

        LinearRegression linearRegression = new LinearRegression();
        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();

        ParamMap[] paramMaps = paramGridBuilder.addGrid(linearRegression.regParam(), new double[] {0.001, 0.01, 0.1, 0.3, 0.5, 0.7,1})
                .addGrid(linearRegression.elasticNetParam(), new double[]{0,0.5,1})
                .build();

        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(linearRegression)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMaps).setTrainRatio(0.9);
        TrainValidationSplitModel model = trainValidationSplit.fit(trainingAndTestDataSet);
        LinearRegressionModel linearRegressionModel = (LinearRegressionModel) model.bestModel();

        linearRegressionModel.transform(holdOutDataSet).filter("label != 0").show();

        System.out.println("The training data r2 value is: " + linearRegressionModel.summary().r2());
        System.out.println("The training data RMSE value is: " + linearRegressionModel.summary().rootMeanSquaredError());

        System.out.println("The holdout data r2 value is: " + linearRegressionModel.evaluate(holdOutDataSet).r2());
        System.out.println("The holdout data RMSE value is: " + linearRegressionModel.evaluate(holdOutDataSet).rootMeanSquaredError());

        System.out.println("Co-efficients:" + linearRegressionModel.coefficients());
        System.out.println("Intercept:" + linearRegressionModel.intercept());
        System.out.println("RegParam:" + linearRegressionModel.getRegParam());
        System.out.println("ElasticNetParam:" + linearRegressionModel.getElasticNetParam());

        System.out.println( "Time taken: " +  (System.nanoTime() -startTime)/1000000 ) ;
    }

}
