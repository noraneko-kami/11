
function exportReport(fileId, format) {
    console.log(`Exporting report for ${fileId} in ${format} format.`);
    window.open(`/api/export/${fileId}?format=${format}`, '_blank');
}

function copyRevisedClause(clauseText) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(clauseText).then(() => {
            showToast('修改后条款已复制到剪贴板', 'success');
        }).catch(err => {
            console.error('复制失败:', err);
            showToast('复制失败，请手动选择文本', 'error');
        });
    } else {
        const textArea = document.createElement('textarea');
        textArea.value = clauseText;
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            showToast('修改后条款已复制到剪贴板', 'success');
        } catch (err) {
            showToast('复制失败，请手动选择文本', 'error');
        }
        document.body.removeChild(textArea);
    }
}

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type === 'success' ? 'success' : type === 'error' ? 'danger' : 'info'} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 5000);
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    container.style.zIndex = '1055';
    document.body.appendChild(container);
    return container;
}

function toggleImplementationSteps(button) {
    const stepsContainer = button.nextElementSibling;
    const isHidden = stepsContainer.style.display === 'none';
    
    stepsContainer.style.display = isHidden ? 'block' : 'none';
    button.innerHTML = isHidden ? 
        '<i class="bi bi-chevron-up"></i> 隐藏步骤' : 
        '<i class="bi bi-chevron-down"></i> 显示步骤';
}

function highlightRiskLevel(riskLevel) {
    const accordionItems = document.querySelectorAll('.accordion-item');
    
    accordionItems.forEach(item => {
        const badge = item.querySelector('.badge');
        if (badge && badge.textContent.trim() === riskLevel) {
            item.style.border = '2px solid #007bff';
            item.style.backgroundColor = '#f8f9fa';
        } else {
            item.style.border = '';
            item.style.backgroundColor = '';
        }
    });
}

function expandAllSuggestions() {
    const collapseElements = document.querySelectorAll('.accordion-collapse');
    collapseElements.forEach(collapse => {
        const bsCollapse = new bootstrap.Collapse(collapse, {
            show: true
        });
    });
}

function collapseAllSuggestions() {
    const collapseElements = document.querySelectorAll('.accordion-collapse.show');
    collapseElements.forEach(collapse => {
        const bsCollapse = bootstrap.Collapse.getInstance(collapse);
        if (bsCollapse) {
            bsCollapse.hide();
        }
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    const revisedClauses = document.querySelectorAll('.revised-clause-box');
    revisedClauses.forEach(clause => {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'btn btn-sm btn-outline-success float-end';
        copyBtn.innerHTML = '<i class="bi bi-clipboard"></i> 复制';
        copyBtn.onclick = () => copyRevisedClause(clause.textContent);
        clause.appendChild(copyBtn);
    });
}); 