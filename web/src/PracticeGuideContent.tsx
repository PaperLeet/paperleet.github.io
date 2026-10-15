import { ArrowRight, BookOpen } from "lucide-react";
import { useI18n } from "./i18n";

export default function PracticeGuideContent({onClose}:{onClose:()=>void}) {
  const {t}=useI18n();
  return <>
            <div className="eyebrow">
              <BookOpen size={16} />
              PAPERLEET
            </div>
            <h2 id="guide-title">
              {t("实现论文里的核心模块", "Implement a paper’s core modules")}
            </h2>
            <p>
              {t(
                "阅读论文、编写代码、运行测试都在这个工作区完成。",
                "Read the paper, write code, and run tests in this workspace.",
              )}
            </p>
            <div className="guide-shortcuts"><span><kbd>/</kbd>{t("搜索论文", "Search papers")}</span><span><kbd>⌘ / Ctrl ↵</kbd>{t("运行测试", "Run tests")}</span><span><kbd>Esc</kbd>{t("退出专注或关闭指南", "Exit focus or close guide")}</span></div>
            <ol>
              <li>
                <b>{t("选择论文与模块", "Choose a paper and module")}</b>
                <small>
                  {t(
                    "先了解论文的核心思想和练习范围，再选择模块，阅读接口与约定。",
                    "Read the paper’s core idea and selected scope, then choose a module and review its contract.",
                  )}
                </small>
              </li>
              <li>
                <b>{t("在编辑器中实现", "Implement in the editor")}</b>
                <small>
                  {t(
                    "草稿自动保存；实验台可以帮助你观察公式的行为。",
                    "Drafts are saved automatically. Use the lab to explore how the formulas behave.",
                  )}
                </small>
              </li>
              <li>
                <b>{t("运行全部测试", "Run all tests")}</b>
                <small>
                  {t(
                    "检查失败原因并修改代码。全部通过后，解锁这道题的坑点。",
                    "Read failure details and revise your code. Passing all tests unlocks the problem's notes.",
                  )}
                </small>
              </li>
              <li>
                <b>{t("提交新论文", "Contribute a paper")}</b>
                <small>
                  {t(
                    "提交论文和官方代码来源，生成候选实现与测试；通过验证和人工评审后才会加入题库。",
                    "Submit a paper and official code source. Generated code and tests enter the catalog only after validation and human review.",
                  )}
                </small>
              </li>
            </ol>
            <button className="button" onClick={onClose}>
              {t("开始练习", "Start practicing")} <ArrowRight size={15} />
            </button>
  </>;
}
