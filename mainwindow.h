#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>
#include <QSlider>
#include <QComboBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QImage>
#include <QPixmap>
#include <vector>

using namespace std;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void showMainInterface();
    void loadImage();
    void downloadCompressedImage();
    void goToWelcomePage();
    void performTransmissionExperiment();
    void performTransmissionExperimentWithTwoImages();
    void calculatePSNRAndSSIM();
    void updateCompressionFormat(int index);

private:
    QStackedWidget *stackedWidget;
    QWidget *welcomePage;
    QWidget *mainPage;
    QLabel *originalLabel;
    QLabel *recLabel;
    QLabel *originalCaption;
    QLabel *recCaption;
    QPushButton *loadButton;
    QPushButton *downloadButton;
    QLabel *sizeLabel;
    QLabel *fileSizeLabel;
    QLabel *compressedFileSizeLabel;
    QLabel *psnrLabel;
    QLabel *ssimLabel;
    QPushButton *backButton;
    QSlider *dwtThresholdSlider;
    QSlider *vqClustersSlider;
    QSlider *jpegQualitySlider;
    QLabel *dwtThresholdLabel;
    QLabel *vqClustersLabel;
    QLabel *jpegQualityLabel;
    QComboBox *formatComboBox;
    QLabel *formatLabel;
    QString originalFilePath;

    // Compression parameters
    float dwtThreshold;
    int vqClusters;
    int jpegQuality;
    QString compressionFormat;

    // Helper functions
    void processImage(const QImage& qimage);
    vector<vector<vector<float>>> loadImageData(const QString& filePath);
    QImage vectorToQImage(const vector<vector<vector<float>>>& image);
    void dwt(const vector<vector<float>>& src, vector<vector<float>>& LL, vector<vector<float>>& LH, vector<vector<float>>& HL, vector<vector<float>>& HH);
    vector<vector<float>> vectorQuantization(const vector<vector<float>>& image, int n_clusters) ;
    double calculatePSNR(const vector<vector<float>>& original, const vector<vector<float>>& reconstructed);
    double calculateSSIM(const vector<vector<float>>& original, const vector<vector<float>>& reconstructed);
    vector<vector<float>> inverseDWT(const vector<vector<float>>& LL, const vector<vector<float>>& LH, const vector<vector<float>>& HL, const vector<vector<float>>& HH, int width, int height);
    void simulateTransmission(const QString& filePath, QString& destinationPath, double& transmissionTime);
};

#endif // MAINWINDOW_H
