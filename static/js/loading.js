function initLoading(formId, btnId, loaderId, textId, loadingText) {
    const form = document.getElementById(formId);
    const btn = document.getElementById(btnId);
    const loader = document.getElementById(loaderId);
    const btnText = document.getElementById(textId);

    if (form) {
        form.onsubmit = function() {
            loader.classList.remove('hidden');
            btnText.innerText = loadingText;
            btn.disabled = true;
            btn.classList.add('opacity-75', 'cursor-not-allowed');
        };
    }
}