#include "mainwindow.h"
#include "imagepreviewdialog.h"
#include <QFileDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFileInfo>
#include <QDesktopServices>
#include <QUrl>
#include <QStackedWidget>
#include <QPushButton>
#include <QLabel>
#include <QImage>
#include <QImageReader>
#include <QPixmap>
#include <cmath>
#include <iostream>
#include <QStyle>
#include <QPalette>
#include <QFont>
#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QApplication>
#include <QStyleFactory>
#include <QMessageBox>
#include <chrono>
#include <QScreen>

using namespace std;

// Constructor
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), originalLabel(new QLabel), recLabel(new QLabel),
    originalCaption(new QLabel), recCaption(new QLabel), loadButton(new QPushButton("Load Image")),
    downloadButton(new QPushButton("Download Compressed Image")), sizeLabel(new QLabel),
    fileSizeLabel(new QLabel), compressedFileSizeLabel(new QLabel), psnrLabel(new QLabel),
    ssimLabel(new QLabel), backButton(new QPushButton("Back to Welcome Page")),
    dwtThresholdSlider(new QSlider(Qt::Horizontal)), vqClustersSlider(new QSlider(Qt::Horizontal)),
    jpegQualitySlider(new QSlider(Qt::Horizontal)), dwtThresholdLabel(new QLabel("DWT Threshold: 10")),
    vqClustersLabel(new QLabel("VQ Clusters: 10")), jpegQualityLabel(new QLabel("JPEG Quality: 75")),
    formatComboBox(new QComboBox), formatLabel(new QLabel("Compression Format: JPEG")) {



    // Set up the main window
    QApplication::setStyle(QStyleFactory::create("Windows"));
    setWindowTitle("ADIB HAFIFI Final Year Project");
    setWindowIcon(QIcon(":/img/UTeM-Logo-1.png"));

    // Create a menu bar
    QMenuBar *menuBar = new QMenuBar(this);
    setMenuBar(menuBar);

    // Create a "File" menu
    QMenu *fileMenu = menuBar->addMenu("File");
    QAction *loadAction = fileMenu->addAction("Load Image");
    QAction *exitAction = fileMenu->addAction("Exit");

    // Create a "Transmission" menu
    QMenu *transmissionMenu = menuBar->addMenu("Transmission");
    QAction *transmissionExperimentAction = transmissionMenu->addAction("Transmission Experiment");
    QAction *transmissionExperimentTwoImagesAction = transmissionMenu->addAction("Transmission Experiment with Two Images");

    // Create a "Metrics" menu
    QMenu *metricsMenu = menuBar->addMenu("Metrics");
    QAction *calculatePSNRAndSSIMAction = metricsMenu->addAction("Calculate PSNR and SSIM");

    // Connect menu actions to slots
    connect(loadAction, &QAction::triggered, this, static_cast<void (MainWindow::*)()>(&MainWindow::loadImage));
    connect(exitAction, &QAction::triggered, this, &QApplication::quit);
    connect(transmissionExperimentAction, &QAction::triggered, this, static_cast<void (MainWindow::*)()>(&MainWindow::performTransmissionExperiment));
    connect(transmissionExperimentTwoImagesAction, &QAction::triggered, this, static_cast<void (MainWindow::*)()>(&MainWindow::performTransmissionExperimentWithTwoImages));
    connect(calculatePSNRAndSSIMAction, &QAction::triggered, this, static_cast<void (MainWindow::*)()>(&MainWindow::calculatePSNRAndSSIM));

    // Connect menu actions to slots
    connect(loadAction, &QAction::triggered, this, &MainWindow::loadImage);
    connect(exitAction, &QAction::triggered, this, &QApplication::quit);

    // Central stacked widget
    stackedWidget = new QStackedWidget(this);
    setCentralWidget(stackedWidget);

    /////////////////////////////////////////////////////////////////////// Welcome page
    welcomePage = new QWidget(this);
    QVBoxLayout *welcomeLayout = new QVBoxLayout(welcomePage);
    QLabel *logoLabel = new QLabel(welcomePage);
    QPixmap logo(":/img/UTeM-Logo-1.png");
    logoLabel->setPixmap(logo.scaled(200, 200, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    logoLabel->setAlignment(Qt::AlignCenter);
    QLabel *welcomeLabel = new QLabel("ADIB HAFIFI final year project \n B082110230", welcomePage);
    QPushButton *startButton = new QPushButton("Start Application", welcomePage);
    welcomeLabel->setAlignment(Qt::AlignCenter);
    welcomeLabel->setStyleSheet("QLabel { font-size: 20px; color: #ffffff; }");
    startButton->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; padding: 15px; font-size: 16px; }"
                               "QPushButton:hover { background-color: #45a049; }");
    welcomeLayout->addWidget(logoLabel);
    welcomeLayout->addWidget(welcomeLabel);
    welcomeLayout->addWidget(startButton);
    welcomePage->setLayout(welcomeLayout);

    /////////////////////////////////////////////////////// Main page
    mainPage = new QWidget(this);
    QVBoxLayout *mainLayout = new QVBoxLayout(mainPage);
    QGroupBox *imageGroupBox = new QGroupBox("Images", mainPage);
    QHBoxLayout *imageLayout = new QHBoxLayout;
    QVBoxLayout *originalLayout = new QVBoxLayout;
    QVBoxLayout *recLayout = new QVBoxLayout;
    originalLayout->addWidget(originalLabel);
    originalLayout->addWidget(originalCaption);
    recLayout->addWidget(recLabel);
    recLayout->addWidget(recCaption);
    imageLayout->addLayout(originalLayout);
    imageLayout->addLayout(recLayout);
    imageGroupBox->setLayout(imageLayout);
    QGroupBox *dataGroupBox = new QGroupBox("Image Data", mainPage);
    QVBoxLayout *dataLayout = new QVBoxLayout;
    dataLayout->addWidget(sizeLabel);
    dataLayout->addWidget(fileSizeLabel);
    dataLayout->addWidget(compressedFileSizeLabel);
    dataLayout->addWidget(psnrLabel);
    dataLayout->addWidget(ssimLabel);
    dataLayout->addWidget(downloadButton);
    dataLayout->addWidget(dwtThresholdLabel);
    dataLayout->addWidget(dwtThresholdSlider);
    dataLayout->addWidget(vqClustersLabel);
    dataLayout->addWidget(vqClustersSlider);
    dataLayout->addWidget(jpegQualityLabel);
    dataLayout->addWidget(jpegQualitySlider);
    dataLayout->addWidget(formatLabel);
    dataLayout->addWidget(formatComboBox);
    dataGroupBox->setLayout(dataLayout);
    mainLayout->addWidget(loadButton);
    mainLayout->addWidget(imageGroupBox);
    mainLayout->addWidget(dataGroupBox);
    mainLayout->addWidget(backButton);
    mainPage->setLayout(mainLayout);

    stackedWidget->addWidget(welcomePage);
    stackedWidget->addWidget(mainPage);

    loadButton->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 2px; padding: 2px; font-size: 12px; }"
                               "QPushButton:hover { background-color: #45a049; }");

    downloadButton->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 2px; padding: 2px; font-size: 12px; }"
                              "QPushButton:hover { background-color: #45a049; }");

    backButton->setStyleSheet("QPushButton { background-color: #FF0000; color: white; border-radius: 2px; padding: 2px; font-size: 12px; }"
                              "QPushButton:hover { background-color: #45a049; }");
    // Connect signals to slots
    connect(startButton, &QPushButton::clicked, this, &MainWindow::showMainInterface);
    connect(loadButton, &QPushButton::clicked, this, &MainWindow::loadImage);
    connect(downloadButton, &QPushButton::clicked, this, &MainWindow::downloadCompressedImage);
    connect(backButton, &QPushButton::clicked, this, &MainWindow::goToWelcomePage);

    ///////////////////////////////////////////////////////////combo box

    // Set default compression parameters
    dwtThreshold = 10.0f;
    vqClusters = 10;
    jpegQuality = 75;
    compressionFormat = "JPG";

    // Set up sliders
    dwtThresholdSlider->setRange(0, 100);
    dwtThresholdSlider->setValue(10);
    vqClustersSlider->setRange(1, 100);
    vqClustersSlider->setValue(10);
    jpegQualitySlider->setRange(1, 100);
    jpegQualitySlider->setValue(75);

    // Set up the format combo box
    formatComboBox->addItem("JPEG");
    formatComboBox->addItem("PNG");
    formatComboBox->addItem("BMP");
    formatComboBox->setCurrentText("JPEG");

    // Connect sliders to update labels
    connect(dwtThresholdSlider, &QSlider::valueChanged, this, [this](int value) {
        dwtThreshold = value;
        dwtThresholdLabel->setText(QString("DWT Threshold: %1").arg(value));
    });

    connect(vqClustersSlider, &QSlider::valueChanged, this, [this](int value) {
        vqClusters = value;
        vqClustersLabel->setText(QString("VQ Clusters: %1").arg(value));
    });

    connect(jpegQualitySlider, &QSlider::valueChanged, this, [this](int value) {
        jpegQuality = value;
        jpegQualityLabel->setText(QString("JPEG Quality: %1").arg(value));
    });

    // Connect format combo box to update compression format
    connect(formatComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::updateCompressionFormat);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor
MainWindow::~MainWindow() {}

// Show main interface
void MainWindow::showMainInterface() {
    stackedWidget->setCurrentWidget(mainPage);
}

////////////////////////////////*****************//////////////////////////////////////////*****************//////////////////////////////

//////////////////////////////////////////// Load image function
void MainWindow::loadImage() {
    QString filePath = QFileDialog::getOpenFileName(this, "Open Image", "", "Image Files (*.png *.jpg *.bmp)");
    if (!filePath.isEmpty()) {
        QImage qimage(filePath);
        if (qimage.isNull()) {
            QMessageBox::warning(this, "Error", "Failed to load image: " + filePath);
            return;
        }

        // Display image size
        int width = qimage.width();
        int height = qimage.height();
        sizeLabel->setText(QString("Image Size: %1 x %2").arg(width).arg(height));

        // Display file size
        QFileInfo fileInfo(filePath);
        qint64 fileSize = fileInfo.size();
        fileSizeLabel->setText(QString("File Size: %1 bytes").arg(fileSize));

        // Display the original image
        originalLabel->setPixmap(QPixmap::fromImage(qimage));
        originalCaption->setText("Original Image");

        // Store the original file path for later use
        this->originalFilePath = filePath;

        // Process the image using user-defined parameters
        processImage(qimage);
    } else {
        QMessageBox::warning(this, "Error", "No file selected.");
    }
}


/////////////////////////////// Process image function///////////////////////////////
/// \brief MainWindow::processImage
/// \param qimage
///
void MainWindow::processImage(const QImage& qimage) {
    int rows = qimage.height();
    int cols = qimage.width();
    vector<vector<vector<float>>> image(3, vector<vector<float>>(rows, vector<float>(cols)));

    // Convert QImage to 3D vector (RGB channels)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            QColor color = qimage.pixelColor(j, i);
            image[0][i][j] = color.red();
            image[1][i][j] = color.green();
            image[2][i][j] = color.blue();
        }
    }

    // Perform DWT on all color channels
    vector<vector<float>> LL_R, LH_R, HL_R, HH_R;
    vector<vector<float>> LL_G, LH_G, HL_G, HH_G;
    vector<vector<float>> LL_B, LH_B, HL_B, HH_B;
    dwt(image[0], LL_R, LH_R, HL_R, HH_R);
    dwt(image[1], LL_G, LH_G, HL_G, HH_G);
    dwt(image[2], LL_B, LH_B, HL_B, HH_B);

    // Apply thresholds to wavelet sub-bands for lossy compression
    auto applyAdaptiveThreshold = [](vector<vector<float>>& subBand, float threshold) {
        for (auto& row : subBand) {
            for (auto& value : row) {
                if (abs(value) < threshold) {
                    value = 0;
                }
            }
        }
    };

 applyAdaptiveThreshold(LH_R, dwtThreshold);
 applyAdaptiveThreshold(HL_R, dwtThreshold);
 applyAdaptiveThreshold(HH_R, dwtThreshold);
 applyAdaptiveThreshold(LH_G, dwtThreshold);
 applyAdaptiveThreshold(HL_G, dwtThreshold);
 applyAdaptiveThreshold(HH_G, dwtThreshold);
 applyAdaptiveThreshold(LH_B, dwtThreshold);
 applyAdaptiveThreshold(HL_B, dwtThreshold);
 applyAdaptiveThreshold(HH_B, dwtThreshold);

    // Perform Vector Quantization on all color channels
    vector<vector<float>> quantized_LL_R = vectorQuantization(LL_R, vqClusters);
    vector<vector<float>> quantized_LH_R = vectorQuantization(LH_R, vqClusters);
    vector<vector<float>> quantized_HL_R = vectorQuantization(HL_R, vqClusters);
    vector<vector<float>> quantized_HH_R = vectorQuantization(HH_R, vqClusters);
    vector<vector<float>> quantized_LL_G = vectorQuantization(LL_G, vqClusters);
    vector<vector<float>> quantized_LH_G = vectorQuantization(LH_G, vqClusters);
    vector<vector<float>> quantized_HL_G = vectorQuantization(HL_G, vqClusters);
    vector<vector<float>> quantized_HH_G = vectorQuantization(HH_G, vqClusters);
    vector<vector<float>> quantized_LL_B = vectorQuantization(LL_B, vqClusters);
    vector<vector<float>> quantized_LH_B = vectorQuantization(LH_B, vqClusters);
    vector<vector<float>> quantized_HL_B = vectorQuantization(HL_B, vqClusters);
    vector<vector<float>> quantized_HH_B = vectorQuantization(HH_B, vqClusters);

    // Reconstruct the image using inverse DWT on all color channels
    vector<vector<float>> rec_image_R = inverseDWT(quantized_LL_R, quantized_LH_R, quantized_HL_R, quantized_HH_R, cols, rows);
    vector<vector<float>> rec_image_G = inverseDWT(quantized_LL_G, quantized_LH_G, quantized_HL_G, quantized_HH_G, cols, rows);
    vector<vector<float>> rec_image_B = inverseDWT(quantized_LL_B, quantized_LH_B, quantized_HL_B, quantized_HH_B, cols, rows);

    // Combine the reconstructed color channels
    vector<vector<vector<float>>> rec_image = {rec_image_R, rec_image_G, rec_image_B};

    // Convert the reconstructed image back to QImage
    QImage recQImage = vectorToQImage(rec_image);

    // Save the reconstructed image with the selected format and quality
    QString compressedFilePath = "compressed_image." + compressionFormat.toLower();
    recQImage.save(compressedFilePath, compressionFormat.toStdString().c_str(), jpegQuality);

    // Display compressed image file size
    QFileInfo compressedFileInfo(compressedFilePath);
    qint64 compressedFileSize = compressedFileInfo.size();
    recCaption->setText(QString("Compressed Image\nFile Size: %1 bytes").arg(compressedFileSize));
    compressedFileSizeLabel->setText(QString("Compressed File Size: %1 bytes").arg(compressedFileSize));

    // Calculate and display PSNR and SSIM
    double psnr = calculatePSNR(image[0], rec_image[0]);
    double ssim = calculateSSIM(image[0], rec_image[0]);
    psnrLabel->setText(QString("PSNR: %1 dB").arg(psnr));
    ssimLabel->setText(QString("SSIM: %1").arg(ssim));

    // Display the reconstructed image
    recLabel->setPixmap(QPixmap::fromImage(recQImage));

    qDebug() << "Compressed file path:" << compressedFilePath;
    qDebug() << "Compressed file size:" << compressedFileSize << "bytes";
}


///////////////////////////////////////////////////////////////////////////////////////
///
///

ImagePreviewDialog::ImagePreviewDialog(const QPixmap& image, QWidget* parent)
    : QDialog(parent), image(image) {
    setWindowTitle("Image Preview");
    setModal(true);  // Make the dialog modal

    // Create the image label
    imageLabel = new QLabel(this);
    imageLabel->setAlignment(Qt::AlignCenter);

    // Scale the image to fit the screen if it's too large
    QScreen* screen = QApplication::primaryScreen();
    QRect screenGeometry = screen->availableGeometry();
    int maxWidth = screenGeometry.width() * 0.8;  // 80% of screen width
    int maxHeight = screenGeometry.height() * 0.8;  // 80% of screen height

    QPixmap scaledImage = image;
    if (image.width() > maxWidth || image.height() > maxHeight) {
        scaledImage = image.scaled(maxWidth, maxHeight, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    // Set the scaled image to the label
    imageLabel->setPixmap(scaledImage);

    // Resize the dialog to fit the scaled image
    resize(scaledImage.size());

    // Create the save button
    saveButton = new QPushButton("Save Image", this);
    connect(saveButton, &QPushButton::clicked, this, &ImagePreviewDialog::saveImage);

    // Create the close button
    closeButton = new QPushButton("Close", this);
    connect(closeButton, &QPushButton::clicked, this, &QDialog::close);

    // Layout
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(imageLabel);
    layout->addWidget(saveButton);
    layout->addWidget(closeButton);
    setLayout(layout);
}

void ImagePreviewDialog::saveImage() {
    QString saveFilePath = QFileDialog::getSaveFileName(
        this,
        "Save Image",
        QDir::homePath() + "/compressed_image.jpg",
        "Images (*.png *.jpg *.bmp)"
        );

    if (saveFilePath.isEmpty()) {
        return;  // User canceled the dialog
    }

    // Save the image to the selected location
    if (image.save(saveFilePath)) {
        QMessageBox::information(this, "Success", "Image saved successfully.");
    } else {
        QMessageBox::warning(this, "Error", "Failed to save the image.");
    }
}

// Download compressed image
void MainWindow::downloadCompressedImage() {
    // Path to the compressed image
    QString compressedFilePath = "compressed_image." + compressionFormat.toLower();

    // Check if the compressed image exists
    if (!QFile::exists(compressedFilePath)) {
        QMessageBox::warning(this, "Error", "No compressed image available to download.");
        return;
    }

    // Load the compressed image into a QPixmap
    QPixmap compressedImage(compressedFilePath);
    if (compressedImage.isNull()) {
        QMessageBox::warning(this, "Error", "Failed to load the compressed image.");
        return;
    }

    // Open the preview window
    ImagePreviewDialog* previewDialog = new ImagePreviewDialog(compressedImage, this);
    previewDialog->exec();  // Show the dialog as a modal window
}

// Go to welcome page
void MainWindow::goToWelcomePage() {
    stackedWidget->setCurrentWidget(welcomePage);
}

// Update compression format
void MainWindow::updateCompressionFormat(int index) {
    compressionFormat = formatComboBox->itemText(index);
    formatLabel->setText(QString("Compression Format: %1").arg(compressionFormat));
}

///////////////////////////////////////**************************/////////////////////////////////**************************//////////////





////////////////////////////////////////// Perform Discrete Wavelet Transform
void MainWindow::dwt(const vector<vector<float>>& src, vector<vector<float>>& LL, vector<vector<float>>& LH, vector<vector<float>>& HL, vector<vector<float>>& HH) {
    int rows = src.size() / 2;
    int cols = src[0].size() / 2;
    LL.resize(rows, vector<float>(cols));
    LH.resize(rows, vector<float>(cols));
    HL.resize(rows, vector<float>(cols));
    HH.resize(rows, vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float a = src[2 * i][2 * j];
            float b = src[2 * i][2 * j + 1];
            float c = src[2 * i + 1][2 * j];
            float d = src[2 * i + 1][2 * j + 1];
            LL[i][j] = (a + b + c + d) / 4;
            LH[i][j] = (a - b + c - d) / 4;
            HL[i][j] = (a + b - c - d) / 4;
            HH[i][j] = (a - b - c + d) / 4;
        }
    }
}

float calculateAdaptiveThreshold(const vector<vector<float>>& subBand, float baseThreshold) {
    float energy = 0;
    for (const auto& row : subBand) {
        for (float value : row) {
            energy += value * value;
        }
    }
    energy = sqrt(energy / (subBand.size() * subBand[0].size()));  // Normalize energy
    return baseThreshold * energy;
}

void applyAdaptiveThreshold(vector<vector<float>>& subBand, float baseThreshold) {
    float adaptiveThreshold = calculateAdaptiveThreshold(subBand, baseThreshold);
    for (auto& row : subBand) {
        for (auto& value : row) {
            if (abs(value) < adaptiveThreshold) {
                value = 0;
            }
        }
    }
}


/////////////////// Inverse DWT //////////////////////////////////////////////////
vector<vector<float>> MainWindow::inverseDWT(const vector<vector<float>>& LL, const vector<vector<float>>& LH, const vector<vector<float>>& HL, const vector<vector<float>>& HH, int width, int height) {
    int rows = LL.size();
    int cols = LL[0].size();
    vector<vector<float>> rec_image(2 * rows, vector<float>(2 * cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float a = LL[i][j];
            float b = LH[i][j];
            float c = HL[i][j];
            float d = HH[i][j];

            // Reconstruct the original image from the DWT coefficients
            rec_image[2 * i][2 * j] = a + b + c + d;
            rec_image[2 * i][2 * j + 1] = a - b + c - d;
            rec_image[2 * i + 1][2 * j] = a + b - c - d;
            rec_image[2 * i + 1][2 * j + 1] = a - b - c + d;
        }
    }

    return rec_image;
}



//////////////////////////////////////////////  VQ   ////////////////////////////////////////////////

void initializeCentersKMeansPP(vector<float>& centers, const vector<float>& samples, int n_clusters) {
    centers[0] = samples[rand() % samples.size()];  // Randomly choose the first center
    for (int i = 1; i < n_clusters; ++i) {
        vector<float> distances(samples.size());
        for (size_t j = 0; j < samples.size(); ++j) {
            float minDist = abs(samples[j] - centers[0]);
            for (int k = 1; k < i; ++k) {
                float dist = abs(samples[j] - centers[k]);
                if (dist < minDist) {
                    minDist = dist;
                }
            }
            distances[j] = minDist;
        }
        float maxDist = *max_element(distances.begin(), distances.end());
        for (size_t j = 0; j < samples.size(); ++j) {
            distances[j] /= maxDist;  // Normalize
        }
        centers[i] = samples[rand() % samples.size()];
    }
}

vector<vector<float>> MainWindow::vectorQuantization(const vector<vector<float>>& image, int n_clusters) {
    int rows = image.size();
    int cols = image[0].size();
    vector<float> samples(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            samples[i * cols + j] = image[i][j];

    vector<float> centers(n_clusters);
    initializeCentersKMeansPP(centers, samples, n_clusters);

    vector<int> labels(samples.size());
    for (int iter = 0; iter < 10; ++iter) {
        for (vector<float>::size_type i = 0; i < samples.size(); ++i) {
            float min_dist = abs(samples[i] - centers[0]);
            labels[i] = 0;
            for (int j = 1; j < n_clusters; ++j) {
                float dist = abs(samples[i] - centers[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    labels[i] = j;
                }
            }
        }

        vector<float> new_centers(n_clusters, 0);
        vector<int> counts(n_clusters, 0);
        for (vector<float>::size_type i = 0; i < samples.size(); ++i) {
            new_centers[labels[i]] += samples[i];
            counts[labels[i]]++;
        }

        for (int j = 0; j < n_clusters; ++j) {
            if (counts[j] > 0)
                centers[j] = new_centers[j] / counts[j];
        }
    }

    vector<vector<float>> new_image(rows, vector<float>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            new_image[i][j] = centers[labels[i * cols + j]];

    return new_image;
}
////////////////////////////////////////////////////////////////////////////////////
///
///
///



///////////////////////////////////// CALCULATION ////////////////////////////////////////////
// Calculate PSNR
double MainWindow::calculatePSNR(const vector<vector<float>>& original, const vector<vector<float>>& reconstructed) {
    if (original.empty() || reconstructed.empty()) {
        cerr << "Error: One or both images are empty." << endl;
        return -1.0;
    }
    int rows = original.size();
    int cols = original[0].size();
    if (rows != reconstructed.size() || cols != reconstructed[0].size()) {
        cerr << "Error: Image dimensions do not match." << endl;
        return -1.0;
    }
    double mse = 0.0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double diff = original[i][j] - reconstructed[i][j];
            mse += diff * diff;
        }
    }
    mse /= (rows * cols);
    if (mse == 0) {
        return numeric_limits<double>::infinity();
    }
    double psnr = 10 * log10((255 * 255) / mse);
    return psnr;
}

// Calculate SSIM
double MainWindow::calculateSSIM(const vector<vector<float>>& original, const vector<vector<float>>& reconstructed) {
    if (original.empty() || reconstructed.empty()) {
        cerr << "Error: One or both images are empty." << endl;
        return -1.0;
    }
    int rows = original.size();
    int cols = original[0].size();
    if (rows != reconstructed.size() || cols != reconstructed[0].size()) {
        cerr << "Error: Image dimensions do not match." << endl;
        return -1.0;
    }
    double C1 = 6.5025, C2 = 58.5225;
    double ssim = 0.0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double mu_x = original[i][j];
            double mu_y = reconstructed[i][j];
            double sigma_x = 0.0;
            double sigma_y = 0.0;
            double sigma_xy = 0.0;
            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    int x = min(max(i + k, 0), rows - 1);
                    int y = min(max(j + l, 0), cols - 1);
                    sigma_x += (original[x][y] - mu_x) * (original[x][y] - mu_x);
                    sigma_y += (reconstructed[x][y] - mu_y) * (reconstructed[x][y] - mu_y);
                    sigma_xy += (original[x][y] - mu_x) * (reconstructed[x][y] - mu_y);
                }
            }
            sigma_x /= 9.0;
            sigma_y /= 9.0;
            sigma_xy /= 9.0;
            double numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2);
            double denominator = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2);
            ssim += numerator / denominator;
        }
    }
    return ssim / (rows * cols);
}





///////////////////////////////////////////// TEST //////////////////////////////////////////////////////////////

// Simulate transmission
void MainWindow::simulateTransmission(const QString& filePath, QString& destinationPath, double& transmissionTime) {
    auto start = std::chrono::high_resolution_clock::now();
    if (!QFile::copy(filePath, destinationPath)) {
        std::cerr << "Error copying file: " << filePath.toStdString() << " to " << destinationPath.toStdString() << std::endl;
        transmissionTime = -1.0;
        return;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    transmissionTime = elapsed.count();
}

// Perform transmission experiment
void MainWindow::performTransmissionExperiment() {
    if (!originalLabel->pixmap() || originalLabel->pixmap().isNull()) {
        QMessageBox::warning(this, "Error", "Please load an image before performing the transmission experiment.");
        return;
    }

    QString originalFilePath = this->originalFilePath;
    if (originalFilePath.isEmpty()) {
        QMessageBox::warning(this, "Error", "No image loaded.");
        return;
    }

    QString destinationDirectory = QFileDialog::getExistingDirectory(this, "Select Destination Directory");
    if (destinationDirectory.isEmpty()) {
        QMessageBox::warning(this, "Error", "Please select a destination directory.");
        return;
    }

    QFileInfo fileInfo(originalFilePath);
    QString destinationFilePath = destinationDirectory + "/" + fileInfo.fileName();

    if (!QFile::copy(originalFilePath, destinationFilePath)) {
        QMessageBox::warning(this, "Error", "Failed to copy the image to the selected directory.");
        return;
    }

    double transmissionTimeOriginal = 0.0, transmissionTimeCompressed = 0.0;
    QString tempDestinationOriginal = destinationDirectory + "/temp_original_image.jpg";
    QString compressedFilePath = "compressed_image." + compressionFormat.toLower();
    QString tempDestinationCompressed = destinationDirectory + "/temp_compressed_image." + compressionFormat.toLower();

    simulateTransmission(destinationFilePath, tempDestinationOriginal, transmissionTimeOriginal);
    simulateTransmission(compressedFilePath, tempDestinationCompressed, transmissionTimeCompressed);

    QMessageBox::information(this, "Transmission Results",
                             QString("Original Image Transmission Time: %1 seconds\n"
                                     "Compressed Image Transmission Time: %2 seconds\n"
                                     "Time Difference: %3 seconds")
                                 .arg(transmissionTimeOriginal, 0, 'f', 6)
                                 .arg(transmissionTimeCompressed, 0, 'f', 6)
                                 .arg(transmissionTimeOriginal - transmissionTimeCompressed, 0, 'f', 6));
}

// Perform transmission experiment with two images
void MainWindow::performTransmissionExperimentWithTwoImages() {
    QString image1Path = QFileDialog::getOpenFileName(this, "Select First Image", "", "Image Files (*.png *.jpg *.bmp)");
    QString image2Path = QFileDialog::getOpenFileName(this, "Select Second Image", "", "Image Files (*.png *.jpg *.bmp)");

    if (image1Path.isEmpty() || image2Path.isEmpty()) {
        QMessageBox::warning(this, "Error", "Please select two images for the experiment.");
        return;
    }

    QString destinationDirectory = QFileDialog::getExistingDirectory(this, "Select Destination Directory");
    if (destinationDirectory.isEmpty()) {
        QMessageBox::warning(this, "Error", "Please select a destination directory.");
        return;
    }

    double transmissionTimeImage1 = 0.0, transmissionTimeImage2 = 0.0;
    QString tempDestinationImage1 = destinationDirectory + "/temp_image1.jpg";
    QString tempDestinationImage2 = destinationDirectory + "/temp_image2.jpg";

    simulateTransmission(image1Path, tempDestinationImage1, transmissionTimeImage1);
    simulateTransmission(image2Path, tempDestinationImage2, transmissionTimeImage2);

    QMessageBox::information(this, "Transmission Results",
                             QString("First Image Transmission Time: %1 seconds\n"
                                     "Second Image Transmission Time: %2 seconds\n"
                                     "Time Difference: %3 seconds")
                                 .arg(transmissionTimeImage1, 0, 'f', 6)
                                 .arg(transmissionTimeImage2, 0, 'f', 6)
                                 .arg(abs(transmissionTimeImage1 - transmissionTimeImage2), 0, 'f', 6));
}

// Calculate PSNR and SSIM
void MainWindow::calculatePSNRAndSSIM() {
    QString image1Path = QFileDialog::getOpenFileName(this, "Select First Image", "", "Image Files (*.png *.jpg *.bmp)");
    QString image2Path = QFileDialog::getOpenFileName(this, "Select Second Image", "", "Image Files (*.png *.jpg *.bmp)");

    if (image1Path.isEmpty() || image2Path.isEmpty()) {
        QMessageBox::warning(this, "Error", "Please select two images for the calculation.");
        return;
    }

    QImage image1(image1Path);
    QImage image2(image2Path);

    if (image1.isNull() || image2.isNull()) {
        QMessageBox::warning(this, "Error", "Failed to load one or both images.");
        return;
    }

    if (image1.size() != image2.size()) {
        QMessageBox::warning(this, "Error", "The selected images must have the same dimensions.");
        return;
    }

    vector<vector<vector<float>>> image1Data = loadImageData(image1Path);
    vector<vector<vector<float>>> image2Data = loadImageData(image2Path);

    double psnr = calculatePSNR(image1Data[0], image2Data[0]);
    double ssim = calculateSSIM(image1Data[0], image2Data[0]);

    QMessageBox::information(this, "PSNR and SSIM Results",
                             QString("PSNR: %1 dB\n"
                                     "SSIM: %2")
                                 .arg(psnr, 0, 'f', 6)
                                 .arg(ssim, 0, 'f', 6));
}

// Load image data
vector<vector<vector<float>>> MainWindow::loadImageData(const QString& filePath) {
    QImage qimage(filePath);
    if (qimage.isNull()) {
        QMessageBox::warning(this, "Error", "Failed to load image: " + filePath);
        return {};
    }

    int rows = qimage.height();
    int cols = qimage.width();
    vector<vector<vector<float>>> image(3, vector<vector<float>>(rows, vector<float>(cols)));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            QColor color = qimage.pixelColor(j, i);
            image[0][i][j] = color.red();
            image[1][i][j] = color.green();
            image[2][i][j] = color.blue();
        }
    }

    return image;
}

QImage MainWindow::vectorToQImage(const vector<vector<vector<float>>>& image) {
    int rows = image[0].size();
    int cols = image[0][0].size();
    QImage qimage(cols, rows, QImage::Format_RGB32);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int r = static_cast<int>(image[0][i][j]);
            int g = static_cast<int>(image[1][i][j]);
            int b = static_cast<int>(image[2][i][j]);

            // Clamp pixel values to [0, 255]
            r = max(0, min(255, r));
            g = max(0, min(255, g));
            b = max(0, min(255, b));

            qimage.setPixel(j, i, qRgb(r, g, b));
        }
    }
    return qimage;
}
