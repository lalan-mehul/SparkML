import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;



public class SparkMLExample {

    public static void main(String[] args) {
        long startTime = System.nanoTime();
        SparkSession sparkSession = SparkSession.builder().appName("TestingSparkML").master("local[*]").getOrCreate();
        Dataset<Row> csvData
                = sparkSession.read().option("header", true)
                                     .option("inferSchema",true)
                                     .csv("src/main/resources/GymCompetition.csv");

        //csvData.printSchema();

        StringIndexer genderIndexer = new StringIndexer();
        genderIndexer.setInputCol("Gender");
        genderIndexer.setOutputCol("GenderIndex");

        csvData = genderIndexer.fit(csvData).transform(csvData);

        //csvData.show();


        OneHotEncoder encoder = new OneHotEncoder();
        encoder.setInputCol("GenderIndex");
        encoder.setOutputCol("GenderVector");

        csvData = encoder.fit(csvData).transform(csvData);
        csvData.show();

        VectorAssembler assembler = new VectorAssembler();
        assembler.setInputCols(new String[] {"Age","Height","Weight","GenderVector"});
        assembler.setOutputCol("features");
        Dataset<Row> csvDataWithFeatures = assembler.transform(csvData);
        //csvDataWithFeatures.show();
        Dataset<Row> modelInputData = csvDataWithFeatures.select("NoOfReps", "features").withColumnRenamed("NoOfReps", "label");
        modelInputData.show();

        LinearRegression linearRegression = new LinearRegression();
        LinearRegressionModel model = linearRegression.fit(modelInputData);
        Vector coefficients = model.coefficients();
        double intercept = model.intercept();

        System.out.println("Model has coefficients:" + coefficients);
        System.out.println("Model has intercept:" + intercept);

        Dataset<Row> modelTransformedData = model.transform(modelInputData);
        modelTransformedData.show();

        System.out.println( "Time taken: " +  (System.nanoTime() -startTime)/1000000 ) ;
    }

}
