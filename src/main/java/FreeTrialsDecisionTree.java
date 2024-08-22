import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.*;

public class FreeTrialsDecisionTree {

    public static UDF1<String,String> countryGrouping = country -> {
        List<String> topCountries =  Arrays.asList("GB","US","IN","UNKNOWN");
        List<String> europeanCountries =  Arrays.asList("BE","BG","CZ","DK","DE","EE","IE","EL","ES","FR","HR","IT","CY","LV","LT","LU","HU","MT","NL","AT","PL","PT","RO","SI","SK","FI","SE","CH","IS","NO","LI","EU");

        if (topCountries.contains(country)) return country;
        if (europeanCountries .contains(country)) return "EUROPE";
        else return "OTHER";
    };

    public static void main (String[] args) {
        long startTime = System.nanoTime();
        SparkSession sparkSession = SparkSession.builder().appName("HousePriceAnalysis").master("local[*]").getOrCreate();
        sparkSession.udf().register("countryGrouping",countryGrouping, DataTypes.StringType);
        Dataset<Row> csvData
                = sparkSession.read().option("header", true)
                .option("inferSchema",true)
                .csv("src/main/resources/vppFreeTrials.csv");
        csvData = csvData
                        .withColumn("country", callUDF("countryGrouping", col("country")))
                        .withColumn("label",when(col("payments_made").geq(1),lit(1)).otherwise(lit(0)));

        StringIndexer countryIndexer = new StringIndexer();
        csvData = countryIndexer.setInputCol("country").setOutputCol("country_index").fit(csvData).transform(csvData);

        new IndexToString()
                .setInputCol("country_index")
                .setOutputCol("value")
                .transform(csvData.select("country_index").distinct())
                .show();

        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler
                .setInputCols(new String[]{"country_index","rebill_period","chapter_access_count","seconds_watched"})
                .setOutputCol("features");
        Dataset<Row> inputData = vectorAssembler
                                            .transform(csvData)
                                            .select("features","label");
        Dataset<Row>[] trainingAndHoldoutData = inputData.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingData = trainingAndHoldoutData[0];
        Dataset<Row> holdOutData = trainingAndHoldoutData[1];

        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier();
        decisionTreeClassifier.setMaxDepth(3);
        DecisionTreeClassificationModel decisionTreeClassificationModel = decisionTreeClassifier.fit(trainingData);
        Dataset<Row> predictions = decisionTreeClassificationModel.transform(holdOutData);
        predictions.show();
        System.out.println(decisionTreeClassificationModel.toDebugString());

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
        evaluator.setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println( "Accuracy of the decision tree is: " + accuracy);

        RandomForestClassifier randomForestClassifier = new RandomForestClassifier();
        randomForestClassifier.setMaxDepth(3);
        RandomForestClassificationModel randomForestClassificationModel = randomForestClassifier.fit(trainingData);
        Dataset<Row> predictions2 = randomForestClassificationModel.transform(holdOutData);
        predictions2.show();
        System.out.println(randomForestClassificationModel.toDebugString());

        accuracy = evaluator.evaluate(predictions2);
        System.out.println( "Accuracy of the random forest is: " + accuracy);

        System.out.println( "Time taken: " +  (System.nanoTime() -startTime)/1000000 ) ;

    }
}
