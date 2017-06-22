//  Predict if a user clicked on an advertisement or not. 
  
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
// Import Pipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val spark = SparkSession.builder().getOrCreate()

val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("logs.csv")

data.printSchema()

//Sample row
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example Data Row")
for(ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}
//    - Rename the Clicked on Ad column to "label"
//    - Grab the following columns "Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Timestamp","Male"

// Hour column from timestamp
val timedata = data.withColumn("Hour",hour(data("Timestamp")))

val logdata = (timedata.select(data("Clicked on Ad").as("label"),
                    $"Daily Time Spent on Site", $"Age", $"Area Income",
                    $"Daily Internet Usage",$"Hour",$"Male"))



val assembler = (new VectorAssembler().setInputCols(Array("Daily Time Spent on Site", "Age", "Area Income",
                  "Daily Internet Usage","Hour")).setOutputCol("features") )


//train test split of 70/30
val Array(training, test) = logdata.randomSplit(Array(0.7, 0.3), seed = 12345)

val lr = new LogisticRegression()
val pipeline = new Pipeline().setStages(Array(assembler, lr))

val model = pipeline.fit(training)
val results = model.transform(test)

// Convert the test results to an RDD using .as and .rdd
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

println("Confusion matrix:")
println(metrics.confusionMatrix)
