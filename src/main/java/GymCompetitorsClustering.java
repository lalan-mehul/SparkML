import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GymCompetitorsClustering {

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

        VectorAssembler assembler = new VectorAssembler()
                                        .setInputCols(new String[]{"GenderVector","Age","Height","Weight","NoOfReps"})
                                        .setOutputCol("features");
        Dataset<Row> inputData = assembler.transform(csvData).select("features");
        KMeans kMeans = new KMeans();
        for (int noOfCluster = 2; noOfCluster<=8;noOfCluster++) {
            kMeans.setK(noOfCluster);
            KMeansModel kMeansModel = kMeans.fit(inputData);
            Dataset<Row> predictions = kMeansModel.transform(inputData);
//          predictions.show();

            Vector[] vectors = kMeansModel.clusterCenters();
//        for (Vector v : vectors) {
//            System.out.println(v);
//        }
            predictions.groupBy("prediction").count().show();

            ClusteringEvaluator evaluator = new ClusteringEvaluator();
            double silhouette = evaluator.evaluate(predictions);
            System.out.println("silhouette with squared euclidean distance: " + silhouette);

            System.out.println(kMeansModel.summary().trainingCost());
            System.out.println( "Time taken: " +  (System.nanoTime() -startTime)/1000000 ) ;

        }
    }

}
