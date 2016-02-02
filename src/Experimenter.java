import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class Experimenter {
    public static void main(String[] args) throws Exception {
        /*
         * Bayad har dataset ba dataset ham name khodesh be surate crossValidation Evaluate beshe va age ham nam
         * nabashan, bayad modelEvaluation anjam bedim. baraye mesal data[1] ba data[1] bayad crossValidation anjam bede
         * va ba data[2] bayad modelValidation anjam bede.
         */

        int countOfSampling = 6;
        Instances trainingData;
        Classifier model;
        Evaluation evaluation = null;

        Instances [] dataArray = readInstance();

        for (int i = 0; i < dataArray.length; i++) {
            trainingData = resampling(dataArray[i], 100);
            for (int j = 0; j < countOfSampling; j++) {
                model = j48Classification(trainingData);

                for (int k = 0; k < dataArray.length; k++) {
                    evaluation = new Evaluation(trainingData);

                    if (k == i)
                        evaluation.crossValidateModel(model, trainingData, 10, new Random(1));
                    else
                        evaluation.evaluateModel(model, dataArray[k]);
                }

                trainingData = resampling(trainingData, 50);
            }
        }

        model = j48Classification(dataArray[1]);
        evaluation = new Evaluation(dataArray[1]);
        evaluation.evaluateModel(model, dataArray[2]);

        System.out.println(evaluation.toSummaryString() + "\n");
        System.out.println(evaluation.toMatrixString() + "\n");
        System.out.println(evaluation.toClassDetailsString() + "\n");
    }

    public static Instances[] readInstance() throws IOException {

        //FileHaye arff dakhele folder bayad be shekle data0.arff ta data4.arff bashad
        Instances[] dataArray = new Instances[5];
        for (int i = 0;i < 5; i++) {
            BufferedReader reader = new BufferedReader(
                    new FileReader("data" + i + ".arff"));
            dataArray[i] = new Instances(reader);
            reader.close();
            // setting class attribute
            dataArray[i].setClassIndex(dataArray[i].numAttributes() - 1);
        }


        return dataArray;
    }

    public static Instances resampling(Instances dataSet, double samplingPercentage) throws Exception {
        Resample filter = new Resample();
        filter.setSampleSizePercent(samplingPercentage);
        filter.setInputFormat(dataSet);
        return Filter.useFilter(dataSet, filter);
    }

    public static Classifier j48Classification(Instances trainSet) throws Exception {
        J48 model = new J48();
        String[] options = weka.core.Utils.splitOptions("-C 0.25 -M 2");
        model.setOptions(options);

        model.buildClassifier(trainSet);
        return model;
    }


}