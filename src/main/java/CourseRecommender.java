import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.List;

import static org.apache.spark.sql.functions.*;

public class CourseRecommender {

    public static void main(String[] args) {
        long startTime = System.nanoTime();
        SparkSession sparkSession = SparkSession.builder().appName("TestingSparkML").master("local[*]").getOrCreate();
        Dataset<Row> csvData
                = sparkSession.read().option("header", true)
                .option("inferSchema",true)
                .csv("src/main/resources/VPPcourseViews.csv");
        csvData = csvData.withColumn("proportionWatched", col("proportionWatched").multiply(100));
        csvData.groupBy("userId").pivot("courseId").sum("proportionWatched").show();

        ALS als = new ALS()
                        .setMaxIter(10)
                        .setRegParam(0.1)
                        .setUserCol("userId")
                        .setItemCol("courseId")
                        .setRatingCol("proportionWatched");
        ALSModel alsModel = als.fit(csvData);
        alsModel.setColdStartStrategy("drop");
        Dataset<Row> recommendedForAllUsers = alsModel.recommendForAllUsers(5);
        List<Row> userRecList = recommendedForAllUsers.takeAsList(5);
        for (Row r : userRecList) {
            int userId = r.getAs(0);
            String recs = r.getAs(1).toString();
            System.out.println("For user " + userId + " we might want to recommend " + recs);
            csvData.filter("userId = " + userId).show();
        }

        Dataset<Row> testData = sparkSession.read().option("header", true)
                .option("inferSchema",true)
                .csv("src/main/resources/courseUsers.csv");
        alsModel.transform(testData).show();
        alsModel.recommendForUserSubset(testData, 5).show();
        //csvData.show();

        System.out.println( "Time taken: " +  (System.nanoTime() -startTime)/1000000 ) ;

    }

}
