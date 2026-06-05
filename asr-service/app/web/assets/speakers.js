/* 说话人管理页（声纹库）：列表 / 改名备注 / 删除 / 降级指引。
 * 体例同 offline.js/stream.js：无构建 IIFE，依赖全局 Vue / naive / AsrCommon。
 * 仅消费 /v2/speakers* 管理 API；登记（上传样本）走 API/curl，不在本页。
 */
(function () {
  'use strict';
  const { ref, reactive, computed, onMounted, h } = Vue;
  const { fmtDate, apiKey, authHeaders, mountApp } = window.AsrCommon;

  const AppBody = {
    setup() {
      const message = naive.useMessage();

      // —— 列表状态 ——
      const rows = reactive([]);
      const loading = ref(false);
      // 降级指引：'' 正常 | need_key | unauthorized | disabled | mismatch | error
      const blocked = ref('');
      const blockedDetail = ref('');

      async function load() {
        loading.value = true;
        blocked.value = '';
        try {
          const r = await fetch('/v2/speakers', { headers: authHeaders() });
          if (r.status === 401) {
            blocked.value = apiKey.value.trim() ? 'unauthorized' : 'need_key';
            return;
          }
          if (r.status === 503) {
            const detail = (await r.json()).detail || '';
            blocked.value = detail === 'model_tag_mismatch' ? 'mismatch' : 'disabled';
            return;
          }
          if (!r.ok) {
            blocked.value = 'error';
            blockedDetail.value = 'HTTP ' + r.status;
            return;
          }
          const data = await r.json();
          rows.length = 0;
          rows.push(...(data.speakers || []));
        } catch (e) {
          blocked.value = 'error';
          blockedDetail.value = String(e);
        } finally {
          loading.value = false;
        }
      }
      onMounted(load);

      const BLOCK_GUIDES = {
        need_key: { type: 'warning', title: '需要 API Key',
          body: '声纹库接口强制鉴权：请点击右上角钥匙图标配置 API Key 后刷新。' },
        unauthorized: { type: 'error', title: 'API Key 无效',
          body: '鉴权失败（401）：请核对右上角配置的 API Key 是否与服务端一致。' },
        disabled: { type: 'warning', title: '声纹库未启用',
          body: '启用方法：服务端开启 enable_speaker 与 enable_speaker_db，且必须配置 api_key（声纹属生物识别信息，不允许无鉴权访问）。' },
        mismatch: { type: 'error', title: '声纹模型版本不一致',
          body: '库内模板与当前引擎 model_tag 不一致：登记/识别已禁用，仅保留查看与删除。处理方式：删除库文件重建，或回退到登记时的引擎版本。' },
        error: { type: 'error', title: '加载失败', body: '' },
      };
      const guide = computed(() => {
        const g = BLOCK_GUIDES[blocked.value];
        return g ? { ...g, body: g.body || blockedDetail.value } : null;
      });

      // —— 编辑（改名 / 备注）——
      const edit = reactive({ show: false, id: '', name: '', note: '', saving: false });
      function openEdit(row) {
        edit.id = row.id;
        edit.name = row.name;
        edit.note = row.note || '';
        edit.show = true;
      }
      async function saveEdit() {
        if (!edit.name.trim()) { message.warning('名称不能为空'); return; }
        edit.saving = true;
        try {
          const r = await fetch('/v2/speakers/' + edit.id, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json', ...authHeaders() },
            body: JSON.stringify({ name: edit.name.trim(), note: edit.note.trim() || null }),
          });
          if (!r.ok) throw new Error((await r.json()).detail || 'HTTP ' + r.status);
          message.success('已保存（后续转写将直接显示新名字）');
          edit.show = false;
          await load();
        } catch (e) {
          message.error('保存失败：' + e.message);
        } finally {
          edit.saving = false;
        }
      }

      // —— 删除（硬删除，不可恢复）——
      async function removeSpeaker(row) {
        try {
          const r = await fetch('/v2/speakers/' + row.id, {
            method: 'DELETE', headers: authHeaders(),
          });
          if (!r.ok) throw new Error((await r.json()).detail || 'HTTP ' + r.status);
          message.success('已删除：' + row.name + '（该说话人在后续转写中退回匿名）');
          await load();
        } catch (e) {
          message.error('删除失败：' + e.message);
        }
      }

      // —— 表格列（render 函数式，Naive UI data-table 体例）——
      const columns = [
        {
          title: '名称', key: 'name',
          render: row => h('div', null, [
            h('span', { style: 'font-weight:600;' }, row.name),
            row.note ? h('div', { style: 'font-size:.78em;opacity:.65;' }, row.note) : null,
          ]),
        },
        {
          title: '来源', key: 'source', width: 96,
          render: row => h(naive.NTag, {
            size: 'small', bordered: false,
            type: row.source === 'auto' ? 'warning' : 'success',
          }, { default: () => (row.source === 'auto' ? '自动登记' : '手动登记') }),
        },
        { title: '模板数', key: 'template_count', width: 80, align: 'center' },
        {
          title: '创建时间', key: 'created_at', width: 170,
          render: row => fmtDate(row.created_at),
        },
        {
          title: '操作', key: 'actions', width: 150, align: 'right',
          render: row => h(naive.NSpace, { justify: 'end', size: 'small' }, {
            default: () => [
              h(naive.NButton, { size: 'tiny', tertiary: true, onClick: () => openEdit(row) },
                { default: () => '改名/备注' }),
              h(naive.NPopconfirm, {
                onPositiveClick: () => removeSpeaker(row),
                positiveText: '删除', negativeText: '取消',
                positiveButtonProps: { type: 'error' },
              }, {
                trigger: () => h(naive.NButton, { size: 'tiny', tertiary: true, type: 'error' },
                  { default: () => '删除' }),
                default: () => '硬删除不可恢复：声纹模板与留存音频将一并清除，该说话人在后续转写中退回匿名。',
              }),
            ],
          }),
        },
      ];

      return { rows, loading, guide, columns, load, edit, saveEdit };
    },
    template: `
      <div style="max-width:980px;margin:0 auto;">
        <n-card :bordered="false" class="panel" size="small">
          <template #header>
            <span class="panel-title"><a-icon name="list" size="15"></a-icon>说话人库</span>
          </template>
          <template #header-extra>
            <n-button size="small" tertiary :loading="loading" @click="load">
              <a-icon name="refresh" size="14" style="margin-right:5px;"></a-icon>刷新
            </n-button>
          </template>

          <n-alert v-if="guide" :type="guide.type" :title="guide.title" :show-icon="true"
                   style="margin-bottom:14px;">{{ guide.body }}</n-alert>

          <template v-if="!guide">
            <n-text depth="3" style="display:block;font-size:.8em;margin-bottom:12px;">
              数据永不自动清理；「自动登记」条目来自离线转写的声纹识别（identify_speakers），改名后立即在后续转写中生效。
            </n-text>
            <n-data-table :columns="columns" :data="rows" :loading="loading"
                          :bordered="false" size="small"
                          :row-key="row => row.id">
              <template #empty>
                <n-empty description="暂无说话人：开启 identify_speakers 转写多人音频可自动登记，或经 POST /v2/speakers 手动登记" size="small"></n-empty>
              </template>
            </n-data-table>
          </template>
        </n-card>

        <n-modal v-model:show="edit.show" preset="card" title="改名 / 备注"
                 style="width:380px;" :mask-closable="!edit.saving">
          <n-space vertical size="large">
            <div>
              <n-text depth="3" style="display:block;font-size:.78em;margin-bottom:5px;">显示名称</n-text>
              <n-input v-model:value="edit.name" placeholder="如：张三" maxlength="64"></n-input>
            </div>
            <div>
              <n-text depth="3" style="display:block;font-size:.78em;margin-bottom:5px;">备注（可选）</n-text>
              <n-input v-model:value="edit.note" placeholder="如：产品部，周会常驻" maxlength="200"></n-input>
            </div>
            <n-space justify="end">
              <n-button size="small" :disabled="edit.saving" @click="edit.show = false">取消</n-button>
              <n-button size="small" type="primary" :loading="edit.saving" @click="saveEdit">保存</n-button>
            </n-space>
          </n-space>
        </n-modal>
      </div>`,
  };

  mountApp(AppBody);
})();
