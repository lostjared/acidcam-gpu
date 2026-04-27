#ifndef __METADATA_VIEWER_H_
#define __METADATA_VIEWER_H_

/**
 * @file metadata-viewer.hpp
 * @brief Media metadata viewer dialog. Runs ffprobe on a user-selected file,
 *        parses the JSON output, and presents the result in a tree along with
 *        copy-friendly Markdown / HTML / plain-text summaries suitable for
 *        pasting into a README, web page, or chat message.
 */

#include <QDialog>
#include <QJsonObject>
#include <QString>

class QLineEdit;
class QPushButton;
class QTreeWidget;
class QTreeWidgetItem;
class QPlainTextEdit;
class QTabWidget;
class QProcess;

class MetadataViewer : public QDialog {
    Q_OBJECT
  public:
    explicit MetadataViewer(QWidget *parent = nullptr);

  private slots:
    /// @brief Open a file picker and store the chosen path in the path field.
    void browseFile();
    /// @brief Launch ffprobe against the current file and refresh the tree.
    void analyzeFile();
    /// @brief Copy the currently selected tab's text (Markdown/HTML/Text)
    ///        to the system clipboard.
    void copyCurrentTab();

  private:
    /// @brief Populate the tree widget from a parsed ffprobe JSON object.
    void populateTree(const QJsonObject &root);
    /// @brief Render a Markdown-formatted summary of the metadata.
    QString buildMarkdown(const QJsonObject &root) const;
    /// @brief Render an HTML-formatted summary of the metadata.
    QString buildHtml(const QJsonObject &root) const;
    /// @brief Render a plain-text summary of the metadata.
    QString buildPlainText(const QJsonObject &root) const;
    /// @brief Recursively add a JSON value as a tree item.
    void addJsonValue(QTreeWidgetItem *parent, const QString &key, const QJsonValue &value);
    /// @brief Refresh the copy button label based on the active tab.
    void updateCopyButtonLabel();

    QLineEdit *pathEdit = nullptr;
    QPushButton *browseButton = nullptr;
    QPushButton *analyzeButton = nullptr;
    QPushButton *copyButton = nullptr;
    QPushButton *closeButton = nullptr;
    QTreeWidget *tree = nullptr;
    QTabWidget *tabs = nullptr;
    QPlainTextEdit *markdownPreview = nullptr;
    QPlainTextEdit *htmlPreview = nullptr;
    QPlainTextEdit *textPreview = nullptr;
    int markdownTabIndex = -1;
    int htmlTabIndex = -1;
    int textTabIndex = -1;
};

#endif
