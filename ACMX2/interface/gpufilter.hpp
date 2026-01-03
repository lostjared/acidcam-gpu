#ifndef __GPUFILTER_HPP__
#define __GPUFILTER_HPP__

#include <QDialog>
#include <QComboBox>
#include <QSpinBox>
#include <QListWidget>
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QProcess>
#include <QStringList>
#include <QLineEdit>
#include <QSortFilterProxyModel>
#include <QStandardItemModel>

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
    
    QStandardItemModel *filterModel;
    QSortFilterProxyModel *proxyModel;
  
    QMap<QString, int> filterNameToIndex;
    QStringList filterNames;
};

#endif 
