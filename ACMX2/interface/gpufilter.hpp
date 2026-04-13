#ifndef __GPUFILTER_HPP__
#define __GPUFILTER_HPP__

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QFile>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QProcess>
#include <QPushButton>
#include <QSortFilterProxyModel>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QStringList>
#include <QTextStream>
#include <QVBoxLayout>

class GPUFilterDialog : public QDialog {
    Q_OBJECT
  public:
    explicit GPUFilterDialog(const QString &executablePath, QWidget *parent = nullptr);

    bool isGPUFilterEnabled() const;
    QStringList getSelectedFilterIndices() const;
    int getBufferSize() const;
    QString getFilterArgument() const;

  public slots:
    void addFilter();
    void removeFilter();
    void moveUp();
    void moveDown();
    void clearAll();
    void filterSearchChanged(const QString &text);
    void saveFilterList();
    void loadFilterList();

  private:
    void loadFiltersFromExecutable();
    void setupUI();

    QString execPath;
    QCheckBox *enableCheckBox;
    QComboBox *filterComboBox;
    QLineEdit *searchLineEdit;
    QListWidget *selectedFiltersList;
    QSpinBox *bufferSizeSpinBox;
    QPushButton *addButton;
    QPushButton *removeButton;
    QPushButton *upButton;
    QPushButton *downButton;
    QPushButton *clearButton;
    QPushButton *okButton;
    QPushButton *cancelButton;
    QPushButton *saveButton;
    QPushButton *loadButton;

    QStandardItemModel *filterModel;
    QSortFilterProxyModel *proxyModel;

    QMap<QString, int> filterNameToIndex;
    QStringList filterNames;
};

#endif
