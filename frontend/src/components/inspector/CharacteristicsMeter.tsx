import React from 'react';

interface CharacteristicsMeterProps {
  label: string;
  rating: number; // 0.0 - 5.0 or 0.0 - 1.0 (will scale to 1-5)
  maxDots?: number;
  description?: string;
}

export const CharacteristicsMeter: React.FC<CharacteristicsMeterProps> = ({
  label,
  rating,
  maxDots = 5,
  description,
}) => {
  // Normalize rating between 1 and 5
  const normalized = rating > 1.0 ? Math.min(5, Math.round(rating / 2)) : Math.min(5, Math.max(1, Math.round(rating * 5)));

  return (
    <div className="flex items-center justify-between py-2 border-b border-champagne/10 last:border-0">
      <div>
        <span className="text-xs uppercase tracking-wider text-champagne/80 font-mono block">
          {label}
        </span>
        {description && (
          <span className="text-[10px] text-champagne/40 block mt-0.5 font-light">
            {description}
          </span>
        )}
      </div>

      <div className="flex items-center gap-1.5">
        {Array.from({ length: maxDots }).map((_, index) => {
          const isFilled = index < normalized;
          return (
            <div
              key={index}
              className={`w-2.5 h-2.5 rounded-full transition-all duration-300 ${
                isFilled
                  ? 'bg-vintageGold shadow-[0_0_8px_rgba(212,175,55,0.6)] scale-110'
                  : 'bg-champagne/15'
              }`}
            />
          );
        })}
      </div>
    </div>
  );
};
