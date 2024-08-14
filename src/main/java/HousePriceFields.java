import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class HousePriceFields {

    public static void main(String[] args) {
        long startTime = System.nanoTime();
        SparkSession sparkSession = SparkSession.builder().appName("HousePriceAnalysis").master("local[*]").getOrCreate();
        Dataset<Row> csvData
                = sparkSession.read().option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/kc_house_data.csv");
        csvData.describe().show();

        csvData = csvData.drop("id","date","waterfront","view","condition","grade","yr_renovated","zipcode","lat","long");
        for (String column:csvData.columns()) {
            System.out.println("Co-relation between " + column + " and price is: " + csvData.stat().corr(column,"price"));
        }

        csvData = csvData.drop("sqft_lot","sqft_lot15","yr_built","sqft_living15");
        for (String col1 : csvData.columns()) {
            for (String col2 : csvData.columns()) {
                System.out.println("Co-relation between " + col1 + " and " + col2 + " is: " + csvData.stat().corr(col1,col2));
            }
        }


        System.out.println( "Time taken: " +  (System.nanoTime() -startTime)/1000000 ) ;
    }

}
