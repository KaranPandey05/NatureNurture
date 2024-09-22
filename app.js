function switchLanguage(language) {
    // Select all elements with data attributes
    const elements = document.querySelectorAll('[data-' + language + ']');
    
    elements.forEach(element => {
        // Get the corresponding translation for the chosen language
        const translation = element.getAttribute('data-' + language);
        if (translation) {
            element.innerText = translation;
        }
    });
}