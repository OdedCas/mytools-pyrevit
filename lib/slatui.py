# -*- coding: utf-8 -*-
"""Shared settings dialog for the Stone Screen tools.

pyRevit puts <extension>/lib on sys.path, so scripts can just:

    from slatui import ask_settings

IronPython 2.7 compatible.
"""

import os
import clr
clr.AddReference('System.Windows.Forms')
clr.AddReference('System.Drawing')
import System.Windows.Forms as WF


def settings_path(name):
    return os.path.join(os.environ.get("APPDATA", "C:\\"),
                        "stone_screen_" + name + ".txt")


def _load(path):
    vals = {}
    try:
        if os.path.isfile(path):
            f = open(path, "r")
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    vals[k] = v
            f.close()
    except Exception:
        pass
    return vals


def _save(path, vals):
    try:
        f = open(path, "w")
        for k in vals:
            f.write("{0}={1}\n".format(k, vals[k]))
        f.close()
    except Exception:
        pass


def ask_settings(title, ns, fields, checks=None, combos=None, store=None):
    """Show a settings dialog and write the results back into `ns`.

    title  - window caption
    ns     - the calling script's globals() dict, updated in place
    fields - [(KEY, "label", "float"|"int")]
    checks - [(KEY, "label", true_value, false_value)]
    combos - [(KEY, "label", [option, ...])]
    store  - filename stem for remembering values between runs

    Returns True if the user pressed OK, False on cancel or a bad number.
    """
    checks = checks or []
    combos = combos or []
    path = settings_path(store or "common")
    saved = _load(path)

    rows = len(fields) + len(checks) + len(combos)
    form = WF.Form()
    form.Text = title
    form.Width = 450
    form.Height = 30 * rows + 130
    form.StartPosition = WF.FormStartPosition.CenterScreen
    form.FormBorderStyle = WF.FormBorderStyle.FixedDialog
    form.MaximizeBox = False
    form.MinimizeBox = False

    boxes = {}
    y = 15
    for key, label, kind in fields:
        lab = WF.Label()
        lab.Text = label
        lab.Left = 12
        lab.Top = y + 3
        lab.Width = 300
        form.Controls.Add(lab)
        tb = WF.TextBox()
        tb.Left = 325
        tb.Top = y
        tb.Width = 95
        tb.Text = str(saved.get(key, ns.get(key, "")))
        form.Controls.Add(tb)
        boxes[key] = (tb, kind, label)
        y += 30

    boxes_c = {}
    for key, label, tval, fval in checks:
        cb = WF.CheckBox()
        cb.Text = label
        cb.Left = 12
        cb.Top = y
        cb.Width = 408
        cur = saved.get(key, None)
        if cur is None:
            cb.Checked = (ns.get(key) == tval)
        else:
            cb.Checked = (cur == str(tval))
        form.Controls.Add(cb)
        boxes_c[key] = (cb, tval, fval)
        y += 28

    boxes_m = {}
    for key, label, options in combos:
        lab = WF.Label()
        lab.Text = label
        lab.Left = 12
        lab.Top = y + 3
        lab.Width = 200
        form.Controls.Add(lab)
        cbx = WF.ComboBox()
        cbx.Left = 220
        cbx.Top = y
        cbx.Width = 200
        cbx.DropDownStyle = WF.ComboBoxStyle.DropDownList
        for o in options:
            cbx.Items.Add(o)
        cbx.SelectedItem = saved.get(key, ns.get(key))
        if cbx.SelectedIndex < 0:
            cbx.SelectedIndex = 0
        form.Controls.Add(cbx)
        boxes_m[key] = cbx
        y += 32

    y += 8
    ok = WF.Button()
    ok.Text = "OK"
    ok.Left = 235
    ok.Top = y
    ok.Width = 90
    ok.DialogResult = WF.DialogResult.OK
    form.Controls.Add(ok)
    form.AcceptButton = ok

    cancel = WF.Button()
    cancel.Text = "Cancel"
    cancel.Left = 330
    cancel.Top = y
    cancel.Width = 90
    cancel.DialogResult = WF.DialogResult.Cancel
    form.Controls.Add(cancel)
    form.CancelButton = cancel

    if form.ShowDialog() != WF.DialogResult.OK:
        return False

    out = {}
    for key in boxes:
        tb, kind, label = boxes[key]
        raw = tb.Text.strip().replace(",", ".")
        try:
            val = int(float(raw)) if kind == "int" else float(raw)
        except Exception:
            WF.MessageBox.Show(
                "'{0}' is not a number for:\n{1}".format(raw, label), title)
            return False
        ns[key] = val
        out[key] = val

    for key in boxes_c:
        cb, tval, fval = boxes_c[key]
        ns[key] = tval if cb.Checked else fval
        out[key] = ns[key]

    for key in boxes_m:
        ns[key] = str(boxes_m[key].SelectedItem)
        out[key] = ns[key]

    _save(path, out)
    return True
