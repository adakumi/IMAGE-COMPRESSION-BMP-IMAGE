#ifndef IMAGEPREVIEWDIALOG_H
#define IMAGEPREVIEWDIALOG_H

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

class ImagePreviewDialog : public QDialog {
    Q_OBJECT

public:
    explicit ImagePreviewDialog(const QPixmap& image, QWidget* parent = nullptr);

private:
    QLabel* imageLabel;
    QPushButton* saveButton;
    QPushButton* closeButton;
    QPixmap image;

private slots:
    void saveImage();
};

#endif // IMAGEPREVIEWDIALOG_H
