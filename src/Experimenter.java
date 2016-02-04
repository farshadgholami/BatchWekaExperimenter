import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.io.*;
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
        Classifier j48Model;
        Classifier svmModel;
        Evaluation svmEvaluation;
        Evaluation j48Evaluation;
        BufferedWriter writer;

        Instances [] dataArray = readInstance();

        for (int i = 0; i < dataArray.length; i++) {
            trainingData = resampling(dataArray[i], 100);
            for (int j = 0; j < countOfSampling; j++) {
                j48Model = j48Classification(trainingData);
                svmModel = libSVMClassification(trainingData);

                for (int k = 0; k < dataArray.length; k++) {
                    j48Evaluation = new Evaluation(trainingData);
                    svmEvaluation = new Evaluation(trainingData);
                    if (k == i) {
                        j48Evaluation.crossValidateModel(j48Model, trainingData, 10, new Random(1));
                        svmEvaluation.crossValidateModel(svmModel, trainingData, 10, new Random(1));
                    } else {
                        j48Evaluation.evaluateModel(j48Model, dataArray[k]);
                        svmEvaluation.evaluateModel(svmModel, dataArray[k]);
                    }

                    writer = new BufferedWriter(new FileWriter("j48_trainProportion" + (i + 1) * 10 + "_instancesSize" + trainingData.numInstances() + "_testProportion" + (k + 1) * 10));
                    writer.write(j48Evaluation.toSummaryString() + "\n" + j48Evaluation.toMatrixString() + "\n" + j48Evaluation.toClassDetailsString());
                    writer.close();
                    writer = new BufferedWriter(new FileWriter("svm_trainProportion" + (i + 1) * 10 + "_instancesSize" + trainingData.numInstances() + "_testProportion" + (k + 1) * 10));
                    writer.write(svmEvaluation.toSummaryString() + "\n" + svmEvaluation.toMatrixString() + "\n" + svmEvaluation.toClassDetailsString());
                    writer.close();
                }

                trainingData = resampling(trainingData, 50);
            }
        }

        j48Model = j48Classification(dataArray[1]);
        j48Evaluation = new Evaluation(dataArray[1]);
        j48Evaluation.evaluateModel(j48Model, dataArray[2]);

        System.out.println(j48Evaluation.toSummaryString() + "\n");
        System.out.println(j48Evaluation.toMatrixString() + "\n");
        System.out.println(j48Evaluation.toClassDetailsString() + "\n");
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

    public static Classifier libSVMClassification(Instances trainSet) throws Exception {
        LibSVM model = new LibSVM();
        String[] options = weka.core.Utils.splitOptions("-S 0 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 -seed 1");
        model.setOptions(options);

        model.buildClassifier(trainSet);
        return model;
    }
}