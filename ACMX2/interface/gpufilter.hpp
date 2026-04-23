#ifndef __GPUFILTER_HPP__
#define __GPUFILTER_HPP__

/**
 * @file gpufilter.hpp
 * @brief UI dialog for configuring chained GPU filter indices.
 */

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

/**
 * @brief Dialog that manages optional GPU filter list and buffer settings.
 */
class GPUFilterDialog : public QDialog {
    Q_OBJECT
  public:
    explicit GPUFilterDialog(const QString &executablePath, QWidget *parent = nullptr);

    /// @brief Return whether GPU filtering is enabled.
    bool isGPUFilterEnabled() const;
    /// @brief Return selected filter indices in playback order.
    QStringList getSelectedFilterIndices() const;
    /// @brief Return frame-buffer depth used by filter chain.
    int getBufferSize() const;
    /// @brief Build CLI argument payload for the selected filter state.
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
    void loadUiState();
    void saveUiState();

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
