#ifndef EDITOR_HPP
#define EDITOR_HPP

/**
 * @file editor.hpp
 * @brief Shader text editor widgets with line numbers and GLSL highlighting.
 */

#include "syntax.hpp"
#include <QCloseEvent>
#include <QDialog>
#include <QHash>
#include <QKeyEvent>
#include <QLabel>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPlainTextEdit>
#include <QStatusBar>
#include <QString>
#include <QStringList>
#include <QSyntaxHighlighter>
#include <QVector>
#include <QWidget>

class LineNumberArea;
class QAction;
class QCompleter;
class QMenu;
class QCheckBox;
class QDoubleSpinBox;
class QSlider;
class QScrollArea;
class QTimer;
class QVBoxLayout;
class QStringListModel;

enum class ShaderDiagnosticSeverity { Error,
                                      Warning,
                                      Note };

struct ShaderDiagnostic {
    int line = 1;
    int column = 0;
    ShaderDiagnosticSeverity severity = ShaderDiagnosticSeverity::Error;
    QString message;
};

struct ShaderEditorUniform {
    QString name;
    int slot = -1;
    double minimum = 0.0;
    double maximum = 1.0;
    double step = 0.01;
    double value = 0.0;
};

/**
 * @brief Plain-text editor with line numbers and code-editing helpers.
 */
class CustomTextEdit : public QPlainTextEdit {
    Q_OBJECT
  public:
    explicit CustomTextEdit(QWidget *parent = nullptr);

    /// @brief Paint callback used by the line-number side widget.
    void lineNumberAreaPaintEvent(QPaintEvent *event);
    /// @brief Compute width required for current line-number digit count.
    int lineNumberAreaWidth();
    void updateLineNumberAreaWidth(int newBlockCount);
    /// @brief Replace the line markers and underlines shown for compiler messages.
    void setDiagnostics(const QVector<ShaderDiagnostic> &diagnostics);
    /// @brief Replace the keywords offered by code completion.
    void setCompletionWords(const QStringList &words);

  signals:
    void themeChanged();
    void includeRequested(const QString &includeName);

  protected:
    void changeEvent(QEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

  private slots:
    void updateLineNumberArea(const QRect &rect, int dy);
    void highlightCurrentLine();
    void matchBrackets();

  public:
    void indentSelection();
    void unindentSelection();
    void duplicateLine();
    void moveLineUp();
    void moveLineDown();
    void toggleComment();

  private:
    void autoIndentNewLine();
    void smartHome(bool shift);
    bool hasMultiLineSelection();
    QString completionPrefix() const;
    void insertCompletion(const QString &completion);
    void showCompletionPopup(bool forced);

    LineNumberArea *m_lineNumberArea = nullptr;
    QVector<ShaderDiagnostic> m_diagnostics;
    QCompleter *m_completer = nullptr;
    QStringListModel *m_completionModel = nullptr;
};

class LineNumberArea : public QWidget {
  public:
    explicit LineNumberArea(CustomTextEdit *editor) : QWidget(editor), m_editor(editor) {}

    QSize sizeHint() const override {
        return QSize(m_editor->lineNumberAreaWidth(), 0);
    }

  protected:
    void paintEvent(QPaintEvent *event) override {
        m_editor->lineNumberAreaPaintEvent(event);
    }

  private:
    CustomTextEdit *m_editor;
};

/**
 * @brief Modal shader editor dialog used by ACMX2.
 */
class TextEditor : public QDialog {
    Q_OBJECT

  public:
    explicit TextEditor(QWidget *parent = nullptr);
    /// @brief Replace editor contents and reset displayed text.
    void setText(const QString &text);
    /// @brief Associate editor with a shader file path.
    void setFileName(const QString &filename);
    /// @brief Return the file currently associated with this editor.
    QString fileName() const;
    /// @brief Move to a one-based line and select the requested source match.
    void revealLocation(int lineNumber, int columnNumber = 0, int matchLength = 0);
    /// @brief Mark the associated shader as being compiled.
    void setCompilePending();
    /// @brief Show compiler status and source diagnostics for the associated shader.
    void setCompileResult(bool success, const QString &compilerOutput);
    /// @brief Configure ACMXVK completion and snippets from the active manifest.
    void setShaderContext(bool acmxvk,
                          const QVector<ShaderEditorUniform> &uniforms,
                          const QString &libraryDirectory = QString());
    void setUniformValue(const QString &name, double value);

  signals:
    /// @brief Emitted after the editor successfully writes its contents to disk.
    void fileSaved(const QString &path);
    void openFileRequested(const QString &path, int lineNumber);
    void previewRequested(const QString &path, const QString &source);
    void uniformValueChanged(const QString &name, double value);

  protected:
    void closeEvent(QCloseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

  private:
    void init();
    /// @brief Show a Save/Discard/Cancel prompt if the document is modified.
    /// @return True if the editor may close, false to cancel the close.
    bool maybePromptSave();
    /// @brief Atomically write the document to disk.
    /// @return True when the file was committed successfully.
    bool writeFile(const QString &filePath);
    void saveContents();
    void saveAs();
    void findText();
    void findNext();
    void findPrevious();
    void replaceText();
    void gotoLine();
    void increaseFontSize();
    void decreaseFontSize();
    void resetFontSize();
    void updateFontSize();
    void updateCursorPosition();
    void updateWindowTitle();
    void navigateDiagnostic(int offset);
    QVector<ShaderDiagnostic> parseDiagnostics(const QString &compilerOutput) const;
    void updateDiagnosticActions();
    void updateCompletionWords();
    void insertSnippet(const QString &snippet);
    QString customUniformDefines() const;
    void openInclude(const QString &includeName);
    void requestPreview();
    void revertContents();
    void rebuildUniformControls();

    bool m_modified = false;
    CustomTextEdit *m_textEdit = nullptr;
    GlslSyntaxHighlighter *m_highlighter = nullptr;
    QStatusBar *m_statusBar = nullptr;
    QLabel *m_lineColLabel = nullptr;
    QLabel *m_compileStatusLabel = nullptr;
    QAction *m_nextDiagnosticAction = nullptr;
    QAction *m_previousDiagnosticAction = nullptr;
    QMenu *m_snippetsMenu = nullptr;
    QWidget *m_uniformPanel = nullptr;
    QScrollArea *m_uniformScroll = nullptr;
    QVBoxLayout *m_uniformLayout = nullptr;
    QCheckBox *m_livePreviewCheck = nullptr;
    QTimer *m_previewTimer = nullptr;
    QHash<QString, QDoubleSpinBox *> m_uniformSpins;
    QHash<QString, QSlider *> m_uniformSliders;
    QVector<ShaderDiagnostic> m_diagnostics;
    int m_currentDiagnostic = -1;
    bool m_acmxvkContext = false;
    QVector<ShaderEditorUniform> m_uniforms;
    QString m_libraryDirectory;
    QString filename;
    QString m_lastSearchText;
    int m_fontSize = 24;
};

#endif
