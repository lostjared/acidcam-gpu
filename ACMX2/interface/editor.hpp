#ifndef EDITOR_HPP
#define EDITOR_HPP

/**
 * @file editor.hpp
 * @brief Shader text editor widgets with line numbers and GLSL highlighting.
 */

#include "syntax.hpp"
#include <QCloseEvent>
#include <QDialog>
#include <QKeyEvent>
#include <QLabel>
#include <QPaintEvent>
#include <QPlainTextEdit>
#include <QStatusBar>
#include <QSyntaxHighlighter>
#include <QVector>
#include <QWidget>

class LineNumberArea;

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

  protected:
    void keyPressEvent(QKeyEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

  private slots:
    void updateLineNumberArea(const QRect &rect, int dy);
    void highlightCurrentLine();
    void matchBrackets();

  public:
    void indentSelection();
    void unindentSelection();

  private:
    void autoIndentNewLine();
    void duplicateLine();
    void moveLineUp();
    void moveLineDown();
    void toggleComment();
    void smartHome(bool shift);
    bool hasMultiLineSelection();

    LineNumberArea *m_lineNumberArea = nullptr;
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

  signals:
    /// @brief Emitted after the editor successfully writes its contents to disk.
    void fileSaved(const QString &path);

  protected:
    void closeEvent(QCloseEvent *event) override;

  private:
    void init();
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

    bool m_modified = false;
    CustomTextEdit *m_textEdit = nullptr;
    GlslSyntaxHighlighter *m_highlighter = nullptr;
    QStatusBar *m_statusBar = nullptr;
    QLabel *m_lineColLabel = nullptr;
    QString filename;
    QString m_lastSearchText;
    int m_fontSize = 24;
};

#endif
