#ifndef __SYNTAX__H_
#define __SYNTAX__H_

/**
 * @file syntax.hpp
 * @brief GLSL syntax highlighter for the embedded text editor.
 */

#include <QPalette>
#include <QRegularExpression>
#include <QStringList>
#include <QSyntaxHighlighter>
#include <QTextCharFormat>
#include <QVector>

/**
 * @brief Rule-based GLSL syntax highlighter.
 */
class GlslSyntaxHighlighter : public QSyntaxHighlighter {
    Q_OBJECT
  public:
    explicit GlslSyntaxHighlighter(QTextDocument *parent = nullptr);
    /// @brief Select syntax colors that contrast with the editor palette.
    void setEditorPalette(const QPalette &palette);

  protected:
    void highlightBlock(const QString &text) override;

  private:
    struct HighlightingRule {
        QRegularExpression pattern;
        QTextCharFormat format;
    };

    void initHighlightingRules(bool lightTheme);

    QVector<HighlightingRule> m_highlightingRules;
    QRegularExpression m_commentStartPattern;
    QRegularExpression m_commentEndPattern;
    QTextCharFormat m_multiLineCommentFormat;
};

#endif
